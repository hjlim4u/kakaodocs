import numpy as np
from typing import List, Dict, Optional
import pandas as pd
from collections import Counter, defaultdict
import asyncio
from datetime import datetime

from sqlalchemy import case, extract, func, literal_column, select

from src.database.models import ChatMessage
from .cache_manager import CacheManager
from .db_manager import DatabaseManager

class PatternAnalyzer:
    def __init__(self, db_manager: DatabaseManager, cache_manager: CacheManager):
        self.db_manager = db_manager
        self.cache_manager = cache_manager

    async def analyze_patterns(self, df: pd.DataFrame, chat_id: str = None, user_id: Optional[int] = None) -> Dict:
        """대화 패턴 분석"""
        cached_metrics = None
        if user_id is not None and chat_id is not None:
            cached_metrics = await self._get_cached_metrics(chat_id, user_id)
        
        # 새로운 데이터만 분석
        if cached_metrics:
            df = df[
                (df['datetime'] > cached_metrics['max_datetime']) |
                (df['datetime'] < cached_metrics['min_datetime'])
            ]

        # Counter의 키를 int로 변환
        hourly_counts = Counter(int(h) for h in df['datetime'].dt.hour.values) + Counter((cached_metrics or {}).get('hourly_counts', {}))
        weekday_counts = Counter(int(w) for w in df['datetime'].dt.weekday.values) + Counter((cached_metrics or {}).get('weekday_counts', {}))
        
        patterns = {
            'min_datetime': df['datetime'].min() if not df.empty else (cached_metrics['min_datetime'] if cached_metrics else None),
            'max_datetime': df['datetime'].max() if not df.empty else (cached_metrics['max_datetime'] if cached_metrics else None),
            'message_count': len(df) + (cached_metrics['message_count'] if cached_metrics else 0),
            'hourly_counts': {int(k): v for k, v in hourly_counts.items()},  # int32를 int로 변환
            'weekday_counts': {int(k): v for k, v in weekday_counts.items()},  # int32를 int로 변환
            'peak_hour': int(max(hourly_counts.items(), key=lambda x: x[1])[0]),  # int32를 int로 변환
            'peak_weekday': int(max(weekday_counts.items(), key=lambda x: x[1])[0])  # int32를 int로 변환
        }
        
        # 캐시 갱신
        if user_id is not None:
            await self._update_cache(chat_id, user_id, patterns)
        
        return {
            'daily_patterns': patterns
        }
    
    async def _get_cached_metrics(self, chat_id: str, user_id: int) -> Optional[Dict]:
        cache_key = self.cache_manager.generate_cache_key('pattern_stats', chat_id=chat_id, user_id=user_id)
        cached_metrics = await self.cache_manager.get('pattern_stats', cache_key)
        if cached_metrics:
            return cached_metrics
        """DB에서 기존 분석 결과 조회"""
        async with self.db_manager.async_session() as session:
            query = (
                select(
                    func.min(ChatMessage.datetime).label('min_datetime'),
                    func.max(ChatMessage.datetime).label('max_datetime'),
                    func.count().label('message_count'),
                    # 일별 패턴
                    func.sum(case(
                        (extract('hour', ChatMessage.datetime) == literal_column('hour'), 1),
                        else_=0
                    )).label('hourly_counts'),
                    func.sum(case(
                        (extract('dow', ChatMessage.datetime) == literal_column('weekday'), 1),
                        else_=0
                    )).label('weekday_counts')
                )
                .where(
                    ChatMessage.chat_id == chat_id,
                    ChatMessage.user_id == user_id
                )
            )
            
            result = await session.execute(query)
            metrics = result.first()
            
            return metrics._asdict() if metrics else None

    def _merge_daily_patterns(self, new_patterns: Dict, cached_metrics: Dict) -> Dict:
        """일별 패턴 데이터 병합"""
        merged_hourly = defaultdict(int)
        merged_weekday = defaultdict(int)
        
        # 새로운 데이터 추가
        for hour, count in new_patterns['hourly_activity'].items():
            merged_hourly[hour] += count
        for weekday, count in new_patterns['weekday_activity'].items():
            merged_weekday[weekday] += count
            
        # 캐시된 데이터 추가
        for hour in range(24):
            merged_hourly[hour] += cached_metrics.get(f'hour_{hour}', 0)
        for weekday in range(7):
            merged_weekday[weekday] += cached_metrics.get(f'weekday_{weekday}', 0)
        
        return {
            'hourly_activity': dict(merged_hourly),
            'weekday_activity': dict(merged_weekday),
            'peak_hour': max(merged_hourly.items(), key=lambda x: x[1])[0],
            'peak_weekday': max(merged_weekday.items(), key=lambda x: x[1])[0]
        }


    async def _update_cache(self, chat_id: str, user_id: int, patterns: Dict):
        """패턴 분석 결과 캐시 갱신"""
        cache_key = self.cache_manager.generate_cache_key('pattern_stats', 
                                                         chat_id=chat_id, 
                                                         user_id=user_id)
        self.cache_manager.set('pattern_stats', cache_key, patterns)
