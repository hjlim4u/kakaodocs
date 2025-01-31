from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, func, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    provider = Column(String)  # google, naver, kakao
    provider_id = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime, nullable=True)
    
class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    refresh_token = Column(String, unique=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(String(100), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    sender = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    datetime = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    total_morphemes = Column(Integer, default=0)
    substantives = Column(Integer, default=0)
    predicates = Column(Integer, default=0)
    endings = Column(Integer, default=0)
    modifiers = Column(Integer, default=0)
    expressions = Column(Integer, default=0)
    question = Column(Integer, default=0)
    exclamation = Column(Integer, default=0)
    formal_ending = Column(Integer, default=0)

class ThreadAnalysis(Base):
    __tablename__ = "thread_analyses"
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(String(100), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    turn_taking = Column(JSON)
    response_pattern = Column(JSON)
    conversation_dynamics = Column(JSON)

