import asyncio
from asyncio.log import logger
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from sqlalchemy import select, func, distinct

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
from .db_manager import DatabaseManager
from .overall_stats_analyzer import OverallStatsAnalyzer
from .mongo_manager import MongoManager
from .conversation_thread_analyzer import TurnTakingAnalyzer, ResponsePatternAnalyzer, ConversationDynamicsAnalyzer

class ChatAnalyzer:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.thread_identifier = ThreadIdentifier(self.text_processor)
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        self.interaction_analyzer = InteractionAnalyzer(self.text_processor, self.db_manager, self.cache_manager)
        self.pattern_analyzer = PatternAnalyzer(self.db_manager, self.cache_manager)
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.visualizer = ChatVisualizer()
        self.overall_stats_analyzer = OverallStatsAnalyzer(self.db_manager, self.cache_manager)
        self.thread_analyzer = ConversationThreadAnalyzer([
            TurnTakingAnalyzer(),
            ResponsePatternAnalyzer(),
            ConversationDynamicsAnalyzer()
        ])
    async def analyze_single_chat(self, chat_file: str, user_id: int = None, is_authenticated: bool = True, chat_id: str = None) -> Dict:
        """단일 채팅 파일 분석"""
        # 캐시 확인
        # cache_key = f"analysis:{chat_file}:{is_authenticated}"
        # cached_result = self.cache_manager.get('thread', cache_key)
        # if cached_result is not None:
        #     return cached_result

        # 파일 파싱
        df = await parse_katalk_file(chat_file)
        
        
        # 채팅방 ID 생성 및 참여자 캐싱
        if chat_id is None:
            chat_id = get_chat_id_from_df(df)
        participants = sorted(df['sender'].unique().tolist())
        self.cache_manager.set('participants', chat_id, participants)
        # 메시지 전처리 수행
        df['preprocessed_message'] = df['message'].apply(self.text_processor.preprocess_message)
        
        # 전처리 후 빈 메시지 제거 (미디어 메시지 등)
        # df = df[df['processed_message'].str.len() > 0].copy()
        
        # 원본 메시지 보존 및 데이터베이스 저장
        # df['original_message'] = df['message']
        
        # 전처리된 메시지로 교체
        # df['message'] = df['processed_message']
        # df.drop('processed_message', axis=1, inplace=True)
        
        # 모든 비동기 태스크 생성
        tasks = [
            asyncio.create_task(self.overall_stats_analyzer.analyze_stats(df, chat_id, user_id)),
            asyncio.create_task(self.interaction_analyzer.analyze_interactions(
                df, chat_id, user_id if is_authenticated else None
            )),
            asyncio.create_task(self.pattern_analyzer.analyze_patterns(
                df, chat_id, user_id if is_authenticated else None
            )),
            # asyncio.create_task(self.visualizer.create_timeline(df)),
            # asyncio.create_task(self.visualizer.create_interaction_heatmap(df))
        ]
        
        if is_authenticated:
            
            # 구간 분할 먼저 수행
            thread_results = await self.thread_identifier.identify_threads(df, chat_id)
            threads = thread_results['threads']
            
            # 각 스레드별 분석 태스크 생성
            thread_analysis_tasks = []
            for start_time, end_time in threads:
                thread_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
                thread_tasks = [
                    asyncio.create_task(self.sentiment_analyzer.analyze_sentiment(thread_df, chat_id)),
                    asyncio.create_task(self.thread_analyzer.analyze_thread(thread_df, chat_id))
                ]
                thread_analysis_tasks.extend(thread_tasks)
            
            # 모든 스레드 분석 완료 대기
            thread_results = await asyncio.gather(*thread_analysis_tasks)
            

            thread_analyses = []
            # for i in range(0, len(thread_results), 2):
            #     sentiment_analyses.append(thread_results[i])
            #     thread_analyses.append(thread_results[i + 1])
            for idx, (start_time, end_time) in enumerate(threads):
                thread_result = {
                    'period': {
                        'start_time': start_time,
                        'end_time': end_time
                    },
                    'sentiment': thread_results[idx * 2],
                    'thread': thread_results[idx * 2 + 1]
                }
                thread_analyses.append(thread_result)

            # 전체 분석 결과 구성
            # sentiment_analysis = self._merge_sentiment_analyses(sentiment_analyses)
            # thread_analysis = self._merge_thread_analyses(thread_analyses)

            # 추가 분석 태스크 생성
            # additional_tasks = [
            #     asyncio.create_task(self.sentiment_analyzer.analyze_sentiments(df, chat_id, thread_analysis)),
            #     asyncio.create_task(self.thread_analyzer.analyze_threads(df, chat_id, thread_analysis))
            # ]
            # tasks.extend(additional_tasks)

        # 모든 태스크 실행 및 결과 수집
        results = await asyncio.gather(*tasks)
        
        # if is_authenticated:
        #     basic_stats, interactions, patterns, timeline, heatmap, sentiment_analysis, thread_analysis = results
        # else:
        basic_stats, interactions, patterns = results
        # sentiment_analysis = thread_analysis = None
        

        # 대화 구간 DB 저장 로직
        if is_authenticated and user_id:
            try:
                await self.db_manager.store_thread_analyses(
                    chat_id=chat_id,
                    user_id=user_id,
                    analyses=thread_analyses
                )
            except Exception as e:
                logger.error(f"Failed to store thread analyses: {str(e)}")
        
        # 결과 구성
        result = {
            'chat_id': chat_id,
            'file_name': chat_file,
            'basic_stats': basic_stats,
            'patterns': patterns,
            'interactions': interactions,
            # 'visualizations': {
            #     'timeline': timeline,
            #     'interaction_heatmap': heatmap
            # }
        }
        
        # 인증된 사용자를 위한 추가 결과
        if is_authenticated and thread_analyses:
            result.update({
                'segment_analyses': thread_analyses,
                # 'sentiment_analysis': sentiment_analysis,
                # 'conversation_dynamics': thread_analysis['thread_stats']
            })
        
        # 결과 캐싱
        self.cache_manager.set('thread', cache_key, result)
        return result

    async def analyze_chats(self, chat_files: List[str]) -> Dict[str, Dict]:
        """여러 채팅 파일을 각각 분석"""
        analyses = {}
        for file_path in chat_files:
            analysis = await self.analyze_single_chat(file_path)
            analyses[file_path] = analysis
            
        return analyses 