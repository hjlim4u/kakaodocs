import pandas as pd
import numpy as np
from typing import Dict, Optional
from collections import defaultdict
from sqlalchemy import select, func, distinct
from datetime import datetime
from .cache_manager import CacheManager
from .db_manager import DatabaseManager
from src.database.models import ChatMessage

class OverallStatsAnalyzer:
    def __init__(self, db_manager: DatabaseManager, cache_manager: CacheManager):
        self.db_manager = db_manager
        self.cache_manager = cache_manager

    async def analyze_stats(self, df: pd.DataFrame, chat_id: str = None, user_id: Optional[int] = None) -> Dict:
        """전체 대화의 기초 통계 분석"""
        # 인증된 사용자인 경우 기존 데이터 조회
        cached_stats = None
        if user_id is not None and chat_id is not None:
            cached_stats = await self._get_cached_stats(chat_id, user_id)
        
        # 새로운 데이터만 분석
        new_messages_df = df
        if cached_stats:
            new_messages_df = df[
                (df['datetime'] > cached_stats['max_datetime']) |
                (df['datetime'] < cached_stats['min_datetime'])
            ]

        # 새로운 데이터 분석
        new_stats = self._analyze_new_messages(new_messages_df)
        
        # 기존 데이터와 병합
        if cached_stats:
            return self._merge_stats(new_stats, cached_stats)
        
        # 캐시 갱신
        if user_id is not None:
            stats = {
                'min_datetime': new_messages_df['datetime'].min() if not new_messages_df.empty else cached_stats['min_datetime'],
                'max_datetime': new_messages_df['datetime'].max() if not new_messages_df.empty else cached_stats['max_datetime'],
                'message_count': new_stats['message_count'] + (cached_stats['message_count'] if cached_stats else 0),
                'duration': new_stats['duration'] + (cached_stats['duration'] if cached_stats else 0),
                'messages_per_participant': new_stats['messages_per_participant']
            }
            await self._update_cache(chat_id, user_id, stats)
        
        return new_stats

    def _analyze_new_messages(self, df: pd.DataFrame) -> Dict:
        """새로운 메시지 데이터 분석"""
        participants = sorted(df['sender'].unique().tolist())
        messages_per_participant = df['sender'].value_counts().to_dict()
        
        return {
            'duration': (df['datetime'].max() - df['datetime'].min()).total_seconds(),
            'message_count': len(df),
            'messages_per_participant': messages_per_participant
        }

    async def _get_cached_stats(self, chat_id: str, user_id: int) -> Optional[Dict]:
        """DB 통계 데이터 조회 (캐시 활용)"""
        # 캐시 확인
        cache_key = self.cache_manager.generate_cache_key('overall_stats', chat_id=chat_id, user_id=user_id)
        cached_result = self.cache_manager.get('overall_stats', cache_key)
        if cached_result is not None:
            return cached_result
        
        # DB 조회
        async with self.db_manager.async_session() as session:
            query = (
                select(
                    func.min(ChatMessage.datetime).label('min_datetime'),
                    func.max(ChatMessage.datetime).label('max_datetime'),
                    func.count().label('message_count'),
                    func.array_agg(distinct(ChatMessage.sender)).label('participants'),
                    func.jsonb_object_agg(
                        ChatMessage.sender,
                        func.count()
                    ).label('messages_per_participant')
                )
                .where(
                    ChatMessage.chat_id == chat_id,
                    ChatMessage.user_id == user_id
                )
            )
            
            result = await session.execute(query)
            stats = result.first()
            
            if not stats:
                return None
            
            stats_dict = stats._asdict()
            
            
            return stats_dict

    def _merge_stats(self, new_stats: Dict, cached_stats: Dict) -> Dict:
        """기초 통계 데이터 병합"""
        total_messages = new_stats['message_count'] + cached_stats['message_count']
        

        
        # 사용자별 메시지 수 병합
        merged_messages_per_participant = defaultdict(int)
        for participant, count in new_stats['messages_per_participant'].items():
            merged_messages_per_participant[participant] += count
        for participant, count in cached_stats['messages_per_participant'].items():
            merged_messages_per_participant[participant] += count
        
        # 가중 평균으로 통계 병합
        return {
            'duration': new_stats['duration'] + cached_stats['duration'],
            'message_count': total_messages,
            'messages_per_participant': dict(merged_messages_per_participant)
        }

    async def _update_cache(self, chat_id: str, user_id: int, stats: Dict):
        """통계 분석 결과 캐시 갱신"""
        cache_key = self.cache_manager.generate_cache_key('overall_stats', 
                                                         chat_id=chat_id, 
                                                         user_id=user_id)
        self.cache_manager.set('overall_stats', cache_key, stats) 