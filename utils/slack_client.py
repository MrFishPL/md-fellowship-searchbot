from slack_sdk import WebClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]

# Initialize WebClient for direct messaging and other Slack API calls
client = WebClient(token=SLACK_BOT_TOKEN) 