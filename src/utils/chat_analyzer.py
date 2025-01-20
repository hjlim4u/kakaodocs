import asyncio
import pandas as pd
from typing import List, Dict

from .conversation_thread_analyzer import ConversationThreadAnalyzer
from .interaction_analyzer import InteractionAnalyzer
from .pattern_analyzer import PatternAnalyzer
from .sentiment_analyzer import AdvancedSentimentAnalyzer
from .visualization import ChatVisualizer
from .katalk_parser import parse_katalk_file
from .chat_utils import get_chat_id_from_df
from .cache_manager import CacheManager
from .thread_identifier import ThreadIdentifier
from .text_processor import TextProcessor

class ChatAnalyzer:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.thread_identifier = ThreadIdentifier(self.text_processor)
        self.thread_analyzer = ConversationThreadAnalyzer()
        self.interaction_analyzer = InteractionAnalyzer(self.text_processor)
        self.pattern_analyzer = PatternAnalyzer()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.visualizer = ChatVisualizer()
        self.cache_manager = CacheManager()
    
    async def analyze_single_chat(self, chat_file: str) -> Dict:
        """단일 채팅 파일 분석"""
        # 캐시 확인
        cache_key = f"analysis:{chat_file}"
        cached_result = self.cache_manager.get('thread', cache_key)
        if cached_result is not None:
            return cached_result

        # 파일 파싱
        df = await parse_katalk_file(chat_file)
        
        # 채팅방 ID 생성 및 참여자 캐싱
        chat_id = get_chat_id_from_df(df)
        participants = sorted(df['sender'].unique().tolist())
        self.cache_manager.set('participants', chat_id, participants)
        
        # 구간 분할 먼저 수행
        thread_results = await self.thread_identifier.identify_threads(df, chat_id)
        
        # 패턴 분석은 동기적으로 수행
        patterns = self.pattern_analyzer.analyze_patterns(df)
        
        # 나머지 비동기 분석 태스크 생성
        tasks = [
            asyncio.create_task(self._analyze_overall_stats(df)),
            asyncio.create_task(self.interaction_analyzer.analyze_interactions(df, chat_id, thread_results)),
            asyncio.create_task(self.sentiment_analyzer.analyze_sentiments(df, chat_id, thread_results)),
            asyncio.create_task(self.thread_analyzer.analyze_threads(df, chat_id, thread_results))
        ]
        
        # 비동기 분석 결과 대기
        basic_stats, interactions, sentiment_analysis, thread_analysis = await asyncio.gather(*tasks)
        
        # 시각화 (비동기 처리 불필요)
        visualizations = {
            'timeline': self.visualizer.create_timeline(df),
            'interaction_heatmap': self.visualizer.create_interaction_heatmap(df)
        }
        
        result = {
            'chat_id': chat_id,
            'file_name': chat_file,
            'basic_stats': basic_stats,
            'patterns': patterns,
            'interactions': interactions,
            'segment_analyses': thread_analysis['segments'],
            'sentiment_analysis': sentiment_analysis,
            'conversation_dynamics': thread_analysis['thread_stats'],
            'visualizations': visualizations
        }
        
        # 결과 캐싱
        self.cache_manager.set('thread', cache_key, result)
        return result

    async def _analyze_overall_stats(self, df: pd.DataFrame) -> Dict:
        """전체 대화의 기초 통계 분석"""
        message_lengths = df['message'].str.len()
        std_length = message_lengths.std() if len(df) > 1 else 0.0
        
        chat_id = get_chat_id_from_df(df)
        # 전체 참여자 목록 캐시 활용
        participants = self.cache_manager.get('participants', chat_id) or sorted(df['sender'].unique().tolist())
        
        return {
            'duration': (df['datetime'].max() - df['datetime'].min()).total_seconds(),
            'message_count': len(df),
            'participant_count': len(participants),
            'avg_message_length': message_lengths.mean(),
            'message_length_std': std_length,
            'participants': participants,
            'messages_per_participant': df['sender'].value_counts().to_dict()
        }

    async def analyze_chats(self, chat_files: List[str]) -> Dict[str, Dict]:
        """여러 채팅 파일을 각각 분석"""
        analyses = {}
        for file_path in chat_files:
            analysis = await self.analyze_single_chat(file_path)
            analyses[file_path] = analysis
            
        return analyses 