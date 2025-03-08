import os
from config import DATA_DIR

# Helper function to create directories if they don't exist
def ensure_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR) 