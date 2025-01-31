from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from datetime import datetime, UTC
import pandas as pd
from typing import List, Optional, Dict
from sqlalchemy import and_, or_

from src.config import settings 
from src.database.models import Base, User, UserSession, ChatMessage, ThreadAnalysis

class DatabaseManager:
    def __init__(self, db_url: str = settings.DATABASE_URL):
        self.engine = create_async_engine(db_url, echo=True)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def verify_refresh_token(self, refresh_token: str) -> Optional[dict]:
        """리프레시 토큰 검증 및 사용자 정보 반환"""
        async with self.async_session() as session:
            # 유효한 세션 조회
            query = select(UserSession, User).join(User).where(
                UserSession.refresh_token == refresh_token,
                UserSession.expires_at > datetime.now(UTC)
            )
            result = await session.execute(query)
            session_data = result.first()
            
            if not session_data:
                return None
                
            session, user = session_data
            return {
                "user_id": user.id,
                "email": user.email,
                "session_id": session.id
            }
    
    async def update_refresh_token(self, session_id: int, new_token: str, expires_at: datetime):
        """리프레시 토큰 업데이트"""
        async with self.async_session() as session:
            query = select(UserSession).where(UserSession.id == session_id)
            result = await session.execute(query)
            user_session = result.scalar_one_or_none()
            
            if user_session:
                user_session.refresh_token = new_token
                user_session.expires_at = expires_at
                await session.commit()
                return True
            return False
    
    async def store_chat_messages(self, messages: List[dict]):
        """채팅 메시지들을 데이터베이스에 저장"""
        async with self.async_session() as session:
            # 청크 단위로 분할하여 처리 (메모리 효율성)
            chunk_size = 1000
            for i in range(0, len(messages), chunk_size):
                chunk = messages[i:i + chunk_size]
                chat_messages = [ChatMessage(**msg) for msg in chunk]
                session.add_all(chat_messages)
                await session.flush()  # 현재 청크 데이터베이스에 기록
            
            await session.commit()  # 모든 청크 처리 완료 후 커밋 

    async def store_thread_analyses(self, chat_id: str, user_id: int, analyses: List[Dict]) -> None:
        """대화 구간 분석 결과 저장"""
        async with self.async_session() as session:
            for analysis in analyses:
                # 기존 분석 결과 검색
                query = select(ThreadAnalysis).where(
                    and_(
                        ThreadAnalysis.chat_id == chat_id,
                        ThreadAnalysis.user_id == user_id,
                        or_(
                            ThreadAnalysis.period_start == analysis["period"]["start"],
                            ThreadAnalysis.period_end == analysis["period"]["end"]
                        )
                    )
                )
                result = await session.execute(query)
                existing_analysis = result.scalar_one_or_none()

                if existing_analysis:
                    # 기존 분석 결과 업데이트
                    existing_analysis.period_start = min(existing_analysis.period_start, analysis["period"]["start"])
                    existing_analysis.period_end = max(existing_analysis.period_end, analysis["period"]["end"])
                    existing_analysis.turn_taking = analysis.get("turntaking", {})
                    existing_analysis.response_pattern = analysis.get("responsepattern", {})
                    existing_analysis.conversation_dynamics = analysis.get("conversationdynamics", {})
                else:
                    # 새로운 분석 결과 생성
                    thread_analysis = ThreadAnalysis(
                        chat_id=chat_id,
                        user_id=user_id,
                        period_start=analysis["period"]["start"],
                        period_end=analysis["period"]["end"],
                        turn_taking=analysis.get("turntaking", {}),
                        response_pattern=analysis.get("responsepattern", {}),
                        conversation_dynamics=analysis.get("conversationdynamics", {})
                    )
                    session.add(thread_analysis)

            await session.commit()

    async def get_thread_analyses(self, chat_id: str, user_id: int) -> List[Dict]:
        """특정 채팅방의 대화 구간 분석 결과 조회"""
        async with self.async_session() as session:
            query = select(ThreadAnalysis).where(
                and_(
                    ThreadAnalysis.chat_id == chat_id,
                    ThreadAnalysis.user_id == user_id
                )
            ).order_by(ThreadAnalysis.period_start)
            
            result = await session.execute(query)
            analyses = result.scalars().all()
            
            return [
                {
                    "period": {
                        "start": analysis.period_start,
                        "end": analysis.period_end
                    },
                    "turntaking": analysis.turn_taking,
                    "responsepattern": analysis.response_pattern,
                    "conversationdynamics": analysis.conversation_dynamics
                }
                for analysis in analyses
            ]

    async def get_last_thread_analysis(self, chat_id: str, user_id: int) -> Optional[Dict]:
        """특정 채팅방의 마지막 대화 구간 분석 결과 조회"""
        async with self.async_session() as session:
            query = (
                select(ThreadAnalysis)
                .where(
                    and_(
                        ThreadAnalysis.chat_id == chat_id,
                        ThreadAnalysis.user_id == user_id
                    )
                )
                .order_by(ThreadAnalysis.period_end.desc())
                .limit(1)
            )
            
            result = await session.execute(query)
            analysis = result.scalar_one_or_none()
            
            if not analysis:
                return None
                
            return {
                "period": {
                    "start": analysis.period_start,
                    "end": analysis.period_end
                },
                "turntaking": analysis.turn_taking,
                "responsepattern": analysis.response_pattern,
                "conversationdynamics": analysis.conversation_dynamics
            }

    async def get_messages_in_period(self, chat_id: str, user_id: int, start_time: datetime, end_time: datetime) -> List[Dict]:
        """특정 기간의 채팅 메시지 조회"""
        async with self.async_session() as session:
            query = (
                select(
                    ChatMessage.datetime,
                    ChatMessage.sender,
                    ChatMessage.message
                )
                .where(
                    and_(
                        ChatMessage.chat_id == chat_id,
                        ChatMessage.user_id == user_id,
                        ChatMessage.datetime >= start_time,
                        ChatMessage.datetime <= end_time
                    )
                )
            )
            result = await session.execute(query)
            stored_messages = result.all()
            
            return stored_messages