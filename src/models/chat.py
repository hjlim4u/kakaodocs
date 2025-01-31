from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

class ChatMessage(BaseModel):
    datetime: datetime
    sender: str
    message: str 

class ChatAnalysisResponse(BaseModel):
    chat_id: str
    file_name: str
    basic_stats: Dict
    patterns: Dict
    interactions: Dict
    segment_analyses: List[Dict]
    sentiment_analysis: Dict
    conversation_dynamics: Dict
    visualizations: Optional[Dict] = None

class ErrorResponse(BaseModel):
    detail: str 