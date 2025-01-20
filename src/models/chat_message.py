from datetime import datetime
from pydantic import BaseModel

class ChatMessage(BaseModel):
    datetime: datetime
    sender: str
    message: str 