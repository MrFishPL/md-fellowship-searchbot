import sqlite3
import os
import time
import json
from pathlib import Path
from slack_sdk.errors import SlackApiError
from utils.slack_client import client
from config import CACHE_DB_NAME, DEFAULT_RETRY_AFTER, DATA_DIR

CACHE_DB_PATH = os.path.join(DATA_DIR, CACHE_DB_NAME)

# In-memory cache
users_cache = {}
subteams_cache = {}

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def _initialize_cache_db():
    """
    Initialize the cache database with tables for users and subteams.
    Creates the 'users' and 'subteams' tables if they do not exist.
    """
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        real_name TEXT NOT NULL,
        timestamp INTEGER NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS subteams (
        subteam_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        timestamp INTEGER NOT NULL
    )
    ''')

    conn.commit()
    conn.close()


def _load_cache_from_db():
    """
    Load user and subteam caches from the database into in-memory dictionaries.
    """
    global users_cache, subteams_cache

    if not os.path.exists(CACHE_DB_PATH):
        _initialize_cache_db()
        return

    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, real_name FROM users")
    users_cache = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("SELECT subteam_id, name FROM subteams")
    subteams_cache = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()


def _call_slack_api_with_retry(api_method, **kwargs):
    """
    Helper function to call Slack API with retry logic for rate limits.
    If the Slack API responds with a 429 (Too Many Requests) status code,
    waits for the specified time before retrying.

    Note:
    - When implementing slash commands or interactions that must respond quickly,
      consider returning an immediate 200 response to avoid hitting Slack's timeout.
      See related discussion at:
      [How to avoid Slack command timeout error](https://stackoverflow.com/questions/34896954/how-to-avoid-slack-command-timeout-error).
    """
    while True:
        try:
            return api_method(**kwargs)
        except SlackApiError as e:
            if e.response.get("error") == "ratelimited":
                # Extract retry_after value from headers or use default
                retry_after = int(e.response.headers.get("Retry-After", DEFAULT_RETRY_AFTER))
                print(f"Rate limited. Waiting for {retry_after} seconds...")
                time.sleep(retry_after)
                # Retry the call
                return _call_slack_api_with_retry(api_method, **kwargs)
            else:
                print(f"Error: {e}")
                raise e


def _get_real_name(user_id):
    """
    Get a user's real name from the cache or Slack API.

    Args:
        user_id (str): The user ID to look up

    Returns:
        str: The user's real name or a default string if not found
    """
    if not users_cache:
        _load_cache_from_db()

    if user_id in users_cache:
        return users_cache[user_id]

    try:
        print(f"Fetching user info for {user_id} from Slack API")
        user_info = _call_slack_api_with_retry(client.users_info, user=user_id)
        user_data = user_info.get('user', {})

        real_name = (
            user_data.get('real_name')
            or user_data.get('profile', {}).get('real_name')
            or user_data.get('name')
            or user_data.get('profile', {}).get('display_name')
            or f"Unknown_{user_id}"
        )

        users_cache[user_id] = real_name

        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        current_time = int(time.time())
        cursor.execute(
            "INSERT OR REPLACE INTO users (user_id, real_name, timestamp) VALUES (?, ?, ?)",
            (user_id, real_name, current_time)
        )
        conn.commit()
        conn.close()

        return real_name
    except Exception as e:
        print(f"Error fetching user info for {user_id}: {e}")
        if 'user_info' in locals():
            try:
                print(f"Response content: {json.dumps(user_info, indent=2)}")
            except:
                pass
        return f"User_{user_id}"


def _save_cache_to_db():
    """
    Save the current in-memory cache of users and subteams to the database.
    """
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    current_time = int(time.time())

    for subteam_id, name in subteams_cache.items():
        cursor.execute(
            "INSERT OR REPLACE INTO subteams (subteam_id, name, timestamp) VALUES (?, ?, ?)",
            (subteam_id, name, current_time)
        )

    for user_id, real_name in users_cache.items():
        cursor.execute(
            "INSERT OR REPLACE INTO users (user_id, real_name, timestamp) VALUES (?, ?, ?)",
            (user_id, real_name, current_time)
        )

    conn.commit()
    conn.close()


def _get_subteam_name(subteam_id):
    """
    Get a subteam's name from the cache or Slack API.

    Args:
        subteam_id (str): The subteam ID to look up

    Returns:
        str: The subteam's name or a default string if not found
    """
    if not subteams_cache:
        _load_cache_from_db()

    if not subteam_id:
        return "Unknown"

    if subteam_id in subteams_cache:
        return subteams_cache[subteam_id]

    try:
        print(f"Fetching subteam info for {subteam_id} from Slack API")
        response = _call_slack_api_with_retry(client.usergroups_list)

        usergroup_data = {}
        for usergroup in response.get('usergroups', []):
            if usergroup.get('id') == subteam_id:
                usergroup_data = usergroup
                break

        subteam_name = usergroup_data.get('name', f"Subteam_{subteam_id}")

        subteams_cache[subteam_id] = subteam_name
        _save_cache_to_db()

        return subteam_name
    except Exception as e:
        print(f"Error fetching subteam info for {subteam_id}: {str(e)}")
        return f"Subteam_{subteam_id}"


# Initialize the cache on module import
_initialize_cache_db()
_load_cache_from_db()