from pydantic import BaseModel

# Define a Pydantic model for a single message
class Message(BaseModel):
    content: str
    author_id: str
    author_name: str
    reactions: list
    images: list
    channel_id: str
    channel_name: str
    time_of_publication: str
    thread: list 
    original_text: str
    message_id: str