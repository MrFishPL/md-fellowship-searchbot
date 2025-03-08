# Data directory
DATA_DIR = "data"

# Database configuration
CACHE_DB_NAME = 'slack_cache.sqlite3'

# Regular expression patterns
URL_PATTERN = r'<(https?://[^|>]+)(?:\|[^>]+)?>'
USER_MENTION_PATTERN = r"<@([U][A-Z0-9]+)>"
SUBTEAM_MENTION_PATTERN = r"<!subteam\^([S][A-Z0-9]+)>"

TEMPERATURE = 0

# File paths and directories
JSON_FILENAME = f"{DATA_DIR}/all_channels_messages.json"

# API configuration
DEFAULT_RETRY_AFTER = 10
DEFAULT_API_LIMIT = 200
CHANNELS_LIST_LIMIT = 1000
REQUEST_TIMEOUT = 5
