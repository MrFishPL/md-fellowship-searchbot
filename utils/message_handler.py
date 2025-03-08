import os
import re
import time
import asyncio
import json
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from tqdm import tqdm
from slack_sdk.errors import SlackApiError
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel

# Use explicit names for clarity
from utils.slack_client import client as slack_client
from utils.message_types import Message
from utils.cache_helper import (
    _initialize_cache_db,
    _load_cache_from_db,
    _get_real_name,
    _get_subteam_name,
)

from config.advanced_config import (
    URL_PATTERN,
    USER_MENTION_PATTERN,
    SUBTEAM_MENTION_PATTERN,
    JSON_FILENAME,
    DEFAULT_RETRY_AFTER,
    DEFAULT_API_LIMIT,
    CHANNELS_LIST_LIMIT,
    REQUEST_TIMEOUT,
    DATA_DIR,
    TEMPERATURE,
)
from config.config import (
    ADMIN_EMAILS,
    NOT_ADMIN_MESSAGE,
    NOT_DM_MESSAGE,
    OPENAI_MODEL_NAME,
    START_SEARCH_MESSAGE,
)

# Global OpenAI client (for relevance checking)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

##############################################################################
# Global Vector Store Initialization
##############################################################################
embedding_fn = OpenAIEmbeddings()
vector_store = Chroma(
    collection_name="slack_messages",
    embedding_function=embedding_fn,
    persist_directory=os.path.join(DATA_DIR)
)
# Reference: https://github.com/chroma-core/chroma/issues/2012

##############################################################################
# Helper Functions
##############################################################################
def _call_slack_api(api_method, **kwargs):
    """
    Invoke a Slack API method with retry handling for rate limits.
    Retries indefinitely if rate limited.
    """
    while True:
        try:
            return api_method(**kwargs)
        except SlackApiError as e:
            if e.response.get("error", "") == "ratelimited":
                retry_after = int(e.response.headers.get("Retry-After", DEFAULT_RETRY_AFTER))
                print(f"Rate limited. Waiting for {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"Error: {e}")
                raise

def _replace_user_mentions(text: str) -> str:
    """
    Replace Slack user mentions (e.g. <@U123456>) with their real names.
    """
    for user_id in re.findall(USER_MENTION_PATTERN, text):
        real_name = _get_real_name(user_id)
        text = text.replace(f"<@{user_id}>", real_name)
    return text

def _replace_subteam_mentions(text: str) -> str:
    """
    Replace subteam mentions (e.g. <!subteam^S123456>) with their real names.
    """
    for subteam_id in re.findall(SUBTEAM_MENTION_PATTERN, text):
        subteam_name = _get_subteam_name(subteam_id)
        text = text.replace(f"<!subteam^{subteam_id}>", subteam_name)
    return text

def _process_message_content(text: str) -> str:
    """
    Process message content by replacing both user and subteam mentions.
    """
    text = _replace_user_mentions(text)
    return _replace_subteam_mentions(text)

def _merge_link_attachments(content: str, attachments: list) -> str:
    """
    Merge link attachment details (title, link, description) into the content.
    """
    for attachment in attachments:
        if "title" in attachment and "title_link" in attachment:
            link_title = attachment.get("title", "")
            link_url = attachment.get("title_link", "")
            link_description = attachment.get("text", "")
            if link_description and link_description != "No description available":
                content += f"\n\n*Link Description*: [{link_title}]({link_url})\n_{link_description}_"
    return content

def _extract_time_of_publication(ts_str: str) -> str:
    """
    Convert a Slack timestamp into a human-readable datetime string.
    """
    if not ts_str:
        return ""
    return datetime.fromtimestamp(float(ts_str)).strftime('%Y-%m-%d %H:%M:%S')

def send_direct_message(user_id: str, message: str) -> None:
    """
    Send a direct message to a Slack user.
    """
    try:
        conversation_info = _call_slack_api(slack_client.conversations_open, users=[user_id])
        channel_id = conversation_info["channel"]["id"]
        _call_slack_api(slack_client.chat_postMessage, channel=channel_id, text=message)
        print(f"Direct message sent to user {user_id}")
    except SlackApiError as e:
        print(f"Error sending direct message: {e}")

def _is_user_admin(user_id: str) -> bool:
    """
    Check if a user is an admin based on their email.
    """
    try:
        user_info = _call_slack_api(slack_client.users_info, user=user_id)
        user_email = user_info["user"]["profile"].get("email", "")
        return user_email in ADMIN_EMAILS
    except SlackApiError as e:
        print(f"Error checking admin status: {e}")
        return False

def _is_dm_channel(channel_id: str) -> bool:
    """
    Determine whether the given channel is a direct message channel.
    """
    try:
        channel_info = _call_slack_api(slack_client.conversations_info, channel=channel_id)
        return channel_info["channel"].get("is_im", False)
    except SlackApiError as e:
        print(f"Error checking channel type: {e}")
        return False

##############################################################################
# Vector Store Utility Functions
##############################################################################
def _create_document_text(msg_dict: dict) -> str:
    """
    Create a text representation of the message for indexing.
    Includes details like author, publication time, channel and thread.
    """
    content = msg_dict.get("content", "")
    author_id = msg_dict.get("author_id", "")
    author_name = msg_dict.get("author_name", "")
    channel_id = msg_dict.get("channel_id", "")
    channel_name = msg_dict.get("channel_name", "")
    timestamp = msg_dict.get("time_of_publication", "")
    thread_texts = [
        f"\nThread => {t_msg.get('time_of_publication', '')} by {t_msg.get('author_name', '')}: {t_msg.get('content', '')}"
        for t_msg in msg_dict.get("thread", [])
    ]
    return (
        f"Author: {author_name} (ID: {author_id})\n"
        f"Published: {timestamp}\n"
        f"Channel: {channel_name} (ID: {channel_id})\n"
        f"Content: {content}"
        + "".join(thread_texts)
    )

def _document_exists(msg_dict: dict) -> bool:
    """
    Verify if the message still exists on Slack.
    """
    ch_id = msg_dict.get("channel_id", "")
    ts_str = msg_dict.get("message_id", "")
    if not ch_id or not ts_str:
        return False
    try:
        _call_slack_api(
            slack_client.conversations_history,
            channel=ch_id,
            latest=ts_str,
            inclusive=True,
            limit=1
        )
        return True
    except SlackApiError as e:
        return e.response.get("error", "") != "message_not_found"

##############################################################################
# Slash Command: Sync Public Channels (Asynchronous Approach)
##############################################################################
def _fetch_channel_data(cid: str, cname: str) -> list:
    """
    Fetch all messages from a single channel and return them as a list of message dicts.
    """
    messages_for_channel = []
    try:
        _call_slack_api(slack_client.conversations_join, channel=cid)
        print(f"Joined channel: {cname}")
    except SlackApiError as join_err:
        if join_err.response.get("error", "") == "already_in_channel":
            print(f"Already in channel: {cname}")
        else:
            print(f"Error joining channel {cname}: {join_err.response.get("error", "")}")
            return messages_for_channel

    print(f"Fetching message history for channel: {cname}")
    channel_history = []
    cursor = None
    while True:
        history_response = _call_slack_api(
            slack_client.conversations_history,
            channel=cid,
            cursor=cursor,
            limit=DEFAULT_API_LIMIT
        )
        channel_history.extend(history_response.get("messages", []))
        cursor = history_response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    print(f"Fetched {len(channel_history)} messages from channel: {cname}")
    for msg in reversed(channel_history):
        if not isinstance(msg, dict):
            continue

        original_text = msg.get("text", "")
        content = _process_message_content(original_text)
        author_id = msg.get("user", "")
        author_name = _get_real_name(author_id)
        raw_attachments = msg.get("attachments", [])
        images = [att["image_url"] for att in raw_attachments if "image_url" in att]
        content = _merge_link_attachments(content, raw_attachments)
        ts_str = msg.get("ts", "")
        time_of_publication = _extract_time_of_publication(ts_str)

        # Process any thread responses
        thread = []
        thread_ts = msg.get("thread_ts", "")
        if thread_ts:
            thread_response = _call_slack_api(
                slack_client.conversations_replies,
                channel=cid,
                ts=thread_ts
            )
            for thread_msg in thread_response.get("messages", []):
                if thread_msg.get("ts") == thread_ts:
                    continue
                orig_thread_text = thread_msg.get("text", "")
                t_content = _process_message_content(orig_thread_text)
                t_author_id = thread_msg.get("user", "")
                t_author_name = _get_real_name(t_author_id)
                t_raw_attachments = thread_msg.get("attachments", [])
                t_images = [att["image_url"] for att in t_raw_attachments if "image_url" in att]
                t_content = _merge_link_attachments(t_content, t_raw_attachments)
                t_ts_str = thread_msg.get("ts", "")
                t_time_of_publication = _extract_time_of_publication(t_ts_str)
                thread.append(
                    Message(
                        content=t_content,
                        original_text=orig_thread_text,
                        author_id=t_author_id,
                        author_name=t_author_name,
                        reactions=[],
                        images=t_images,
                        channel_id=cid,
                        channel_name=cname,
                        time_of_publication=t_time_of_publication,
                        thread=[],
                        message_id=t_ts_str
                    ).dict()
                )

        top_level_msg = Message(
            content=content,
            original_text=original_text,
            author_id=author_id,
            author_name=author_name,
            reactions=[],
            images=images,
            channel_id=cid,
            channel_name=cname,
            time_of_publication=time_of_publication,
            thread=thread,
            message_id=ts_str
        ).dict()
        messages_for_channel.append(top_level_msg)

    print(f"Finished processing channel: {cname}")
    return messages_for_channel

def start_search_message(ack, command, say) -> None:
    """
    Slash command handler for /start-search.
    """
    ack()
    user_id = command["user_id"]

    send_direct_message(
        user_id,
        START_SEARCH_MESSAGE
    )
    
def sync_public_channels_messages(ack, command, say) -> None:
    """
    Slash command handler for /sync.
    Fetches messages from public channels and indexes them in the vector store.
    Only admins can run this command via direct message.
    """
    ack()
    user_id = command["user_id"]
    channel_id = command["channel_id"]

    if not _is_dm_channel(channel_id):
        send_direct_message(
            user_id,
            NOT_DM_MESSAGE
        )

        return

    if not _is_user_admin(user_id):
        send_direct_message(
            user_id,
            NOT_ADMIN_MESSAGE
        )
        
        return

    try:
        say("Syncing all public channels. Please wait...")

        # Clear the data directory
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            print(f"Cleared data directory: {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Recreated data directory: {DATA_DIR}")

        # Initialize cache database and reinitialize vector store
        _initialize_cache_db()
        global vector_store
        
        del vector_store
        chromadb.api.client.SharedSystemClient.clear_system_cache()
                
        vector_store = Chroma(
            collection_name="slack_messages",
            embedding_function=embedding_fn,
            persist_directory=DATA_DIR
        )


        channels_list = _call_slack_api(
            slack_client.conversations_list,
            types="public_channel",
            limit=CHANNELS_LIST_LIMIT
        )

        all_texts = []
        all_metadatas = []
        for ch in channels_list.get("channels", []):
            cid = ch["id"]
            cname = ch["name"]
            print(f"Processing channel: {cname} (ID: {cid})")
            messages = _fetch_channel_data(cid, cname)
            for message in messages:
                doc_text = _create_document_text(message)
                all_texts.append(doc_text)
                all_metadatas.append(message)

        print("Vectorizing fetched messages...")
        for doc_text, metadata in tqdm(zip(all_texts, all_metadatas), desc="Embedding Messages", total=len(all_texts)):
            metadata["thread"] = json.dumps(metadata.get("thread", []))
            metadata["reactions"] = json.dumps(metadata.get("reactions", []))
            metadata["images"] = json.dumps(metadata.get("images", []))
            vector_store.add_texts([doc_text], metadatas=[metadata])

        say(f"✅ Successfully synced messages from {len(channels_list.get('channels', []))} channels.")
        print("Vector store updated successfully.")
    except SlackApiError as e:
        say(f"Error fetching channels or messages: {e.response.get("error", "")}")
        print(f"Slack API error: {e.response.get("error", "")}")
    except Exception as e:
        say(f"General error: {str(e)}")
        print(f"General error: {str(e)}")

##############################################################################
# Message Event Handler
##############################################################################
def handle_message(event, say) -> None:
    """
    Process new Slack messages:
      - For public channels: index the message.
      - For direct messages: run a similarity search and, if relevant, forward matching messages.
    """
    # Ignore bot/system messages
    if "bot_id" in event or event.get("subtype", "") in ["bot_message", "channel_join"]:
        return

    user_id = event.get("user")
    if not user_id:
        return

    channel_type = event.get("channel_type")
    if channel_type not in ["channel", "im"]:
        return

    msg_ts = event.get("ts", "")
    text = event.get("text", "")
    processed_content = _process_message_content(text)
    author_name = _get_real_name(user_id)
    channel_id = event.get("channel", "")
    raw_attachments = event.get("attachments", [])
    images = [att["image_url"] for att in raw_attachments if "image_url" in att]
    time_of_publication = _extract_time_of_publication(msg_ts)

    new_message_obj = Message(
        content=processed_content,
        original_text=text,
        author_id=user_id,
        author_name=author_name,
        reactions=[],
        images=images,
        channel_id=channel_id,
        channel_name="",
        time_of_publication=time_of_publication,
        thread=[],
        message_id=msg_ts
    ).dict()

    doc_text = _create_document_text(new_message_obj)
    new_message_obj["thread"] = json.dumps(new_message_obj.get("thread", []))
    new_message_obj["reactions"] = json.dumps(new_message_obj.get("reactions", []))
    new_message_obj["images"] = json.dumps(new_message_obj.get("images", []))

    if channel_type == "channel":
        # Public channel: simply index the message.
        vector_store.add_texts([doc_text], metadatas=[new_message_obj])
        print(f"Added new message {msg_ts} to the vector store.")
        return

    elif channel_type == "im":
        # Direct message: perform similarity search and forward relevant messages.
        found_docs = vector_store.similarity_search(doc_text, k=15)
        if found_docs:
            def check_relevance(doc):
                metadata = getattr(doc, "metadata", {})
                candidate_content = getattr(doc, "page_content", metadata.get("content", ""))
                relevance_prompt = (
                    f"User query: {processed_content}\n"
                    f"Candidate message: {candidate_content}\n"
                    "Is the candidate message truly relevant to the user's query? Answer with 'Yes' or 'No'. "
                    "Pay attention to the author, date, and other specific user requirements."
                )
                try:
                    response = openai_client.chat.completions.create(
                        model=OPENAI_MODEL_NAME,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an assistant that only responds with 'Yes' or 'No'."
                            },
                            {"role": "user", "content": relevance_prompt}
                        ],
                        max_tokens=1,
                        temperature=TEMPERATURE
                    )
                    answer = response.choices[0].message.content.strip().lower()
                    if answer.startswith("yes"):
                        return doc
                except Exception as e:
                    print(f"Error querying OpenAI for relevance: {e}")
                return None

            filtered_docs = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(check_relevance, doc) for doc in found_docs]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        filtered_docs.append(result)

            if filtered_docs:
                _call_slack_api(slack_client.chat_postMessage, channel=channel_id, text="Forwarding relevant messages:")
                for doc in filtered_docs:
                    metadata = getattr(doc, "metadata", {})
                    orig_channel = metadata.get("channel_id", "")
                    orig_ts = metadata.get("message_id", "")
                    author_id = metadata.get("author_id", "")
                    author_name = metadata.get("author_name", "")
                    
                    try:
                        author_image_url = author_image_url.get("image_48", "")
                    except Exception:
                        author_image_url = ""
                        
                    if orig_channel and orig_ts:
                        try:
                            permalink_resp = _call_slack_api(
                                slack_client.chat_getPermalink, channel=orig_channel, message_ts=orig_ts
                            )
                            permalink = permalink_resp.get("permalink", "")
                        except SlackApiError:
                            permalink = None
                        if permalink:
                            msg = f"*{author_name}*\n{metadata.get('original_text', '')}"
                            if len(msg) > 250:
                                msg = msg[:250] + "..."
                                
                            blocks = [
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": ">" + "\n> ".join(msg.split("\n"))
                                    }
                                },
                                {
                                    "type": "actions",
                                    "elements": [
                                        {
                                            "type": "button",
                                            "text": {"type": "plain_text", "text": "View message."},
                                            "url": permalink
                                        }
                                    ]
                                },
                                {"type": "divider"},
                            ]
                            _call_slack_api(
                                slack_client.chat_postMessage,
                                channel=channel_id,
                                blocks=blocks,
                            )
                        else:
                            _call_slack_api(
                                slack_client.chat_postMessage,
                                channel=channel_id,
                                text=f"Could not retrieve permalink for message from {metadata.get('author_name', 'Unknown')}."
                            )
                    else:
                        _call_slack_api(
                            slack_client.chat_postMessage,
                            channel=channel_id,
                            text="Message metadata incomplete; cannot forward."
                        )
            else:
                _call_slack_api(slack_client.chat_postMessage, channel=channel_id, text="No similar messages deemed relevant to forward.")
        else:
            _call_slack_api(slack_client.chat_postMessage, channel=channel_id, text="No similar messages found to forward.")
    return
