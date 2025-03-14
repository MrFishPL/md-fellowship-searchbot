# Slack Bot

A Slack bot that provides channel message syncing and broadcasting capabilities.

## Features

- **Message Sync**: Sync messages from all public channels to a JSON file
- **Semantic search**: Search for messages in your slack workspace
- **Admin Controls**: Restrict command access to administrators only

## Setup

### 1. Environment Variables

Create a `.env` file with your Slack tokens:

```
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
OPENAI_API_KEY=your-openai-api-key-goes-here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Slack App

1. Go to [Slack API Apps page](https://api.slack.com/apps)
2. Select your app (or create a new one)
3. Under "Features" in the sidebar, select "Slash Commands"
4. Click "Create New Command" and add:

For the sync command:
- Command: `/sync`
- Request URL: Your server URL that handles commands (or use Socket Mode)
- Short Description: "Sync messages from all public channels"
- Usage Hint: Just type /sync. Admin only

For the broadcast command:
- Command: `/start-search`
- Request URL: Your server URL that handles commands (or use Socket Mode)
- Short Description: "Starts the conversation"
- Usage Hint: Just type /start-search

5. Save changes

### 4. Permissions

Ensure your bot has the following permissions:
- `channels:history`
- `channels:read`
- `chat:write`
- `commands`
- `im:write`
- `users:read`
- `users:read.email`

### 5. Run the Bot

```bash
python app.py
```

## Usage

- `/sync` - Sync all public channel messages to a JSON file (admin only, DM only)
- `/start-search` - Starts a conversation with a bot

## Configuration

Edit `config/config.py` to update the list of admin emails that are allowed to use administrative commands. # md-fellowship-searchbot
