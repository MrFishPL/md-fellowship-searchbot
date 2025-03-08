from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from dotenv import load_dotenv
from utils.message_handler import (
    send_direct_message, 
    sync_public_channels_messages, 
    handle_message,
    start_search_message,
)
# Load environment variables
load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]

# Initialize Bolt app
app = App(token=SLACK_BOT_TOKEN)

# Register the '/sync' slash command handler
app.command("/sync")(sync_public_channels_messages)
app.command("/start-search")(start_search_message)
# TODO: clear, search

# Register message event listener for non-command messages
app.message()(handle_message)

if __name__ == "__main__":    
    # Start the Socket Mode handler
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start() 