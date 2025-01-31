import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

from .statistical_utils import adjust_outliers
from .text_processor import TextProcessor
from .cache_manager import CacheManager
import asyncio
from .db_manager import DatabaseManager
from collections import defaultdict
from sqlalchemy import func, select
from src.database.models import ChatMessage
import sqlalchemy

class InteractionAnalyzer:
    def __init__(self, text_processor: TextProcessor, db_manager: DatabaseManager, cache_manager: CacheManager):
        self.text_processor = text_processor
        self.db_manager = db_manager
        self.cache_manager = cache_manager


    async def analyze_interactions(self, df: pd.DataFrame, chat_id: str, user_id: Optional[int] = None) -> Dict:
        """대화 상호작용 분석 수행"""
        participants = sorted(df['sender'].unique())

        # 분석 태스크 생성
        style_task = asyncio.create_task(
            self._analyze_language_style(df, participants, chat_id, user_id)
        )
        response_time_task = asyncio.create_task(
            self._analyze_response_times(df, participants)
        )
        
        # 모든 분석 태스크 동시 실행 및 결과 수집
        language_style_results, response_time_results = await asyncio.gather(
            style_task,
            response_time_task
        )
        
        return {
            'language_style_analysis': language_style_results,
            'response_time_analysis': response_time_results
        }

    async def _analyze_language_style(self, df: pd.DataFrame, participants: List[str], chat_id: str, user_id: Optional[int] = None) -> Dict:
        """사용자별 언어 스타일 분석"""
        style_tasks = []
        analyzed_messages = []
        
        for user in participants:
            user_df = df[df['sender'] == user]
            style_tasks.append(self.extract_user_style(user, user_df, chat_id, user_id))
                # DB 저장이 필요한 경우에만 수행
        
        
        style_results = await asyncio.gather(*style_tasks)
        user_styles = {}
        
        # 결과 분리
        for user, style, messages in style_results:
            user_styles[user] = style
            if messages:
                analyzed_messages.extend(messages)
        if user_id is not None:
            await self._store_analyzed_messages(analyzed_messages)

        return {
            'user_styles': user_styles,
            'style_similarities': self._calculate_style_similarities(user_styles),
        }

    async def _get_cached_metrics(self, chat_id: str, user_id: int, sender: str) -> Dict:
        """DB에서 기존 분석 결과의 집계값 조회 (캐시 활용)"""
        cache_key = self.cache_manager.generate_cache_key('interaction_stats', chat_id=chat_id, user_id=user_id, sender=sender)
        cached_result = self.cache_manager.get('interaction_stats', cache_key)
        if cached_result is not None:
            metrics = cached_result
            return metrics
        
        async with self.db_manager.async_session() as session:
            query = (
                select(
                    func.min(ChatMessage.datetime).label('min_datetime'),
                    func.max(ChatMessage.datetime).label('max_datetime'),
                    func.count().label('message_count'),
                    func.sum(ChatMessage.total_morphemes).label('total_morphemes'),
                    func.sum(ChatMessage.substantives).label('total_substantives'),
                    func.sum(ChatMessage.predicates).label('total_predicates'),
                    func.sum(ChatMessage.endings).label('total_endings'),
                    func.sum(ChatMessage.modifiers).label('total_modifiers'),
                    func.sum(ChatMessage.expressions).label('total_expressions'),
                    func.sum(ChatMessage.question).label('total_question'),
                    func.sum(ChatMessage.exclamation).label('total_exclamation'),
                    func.sum(ChatMessage.formal_ending).label('total_formal_ending'),
                )
                .where(
                    ChatMessage.chat_id == chat_id,
                    ChatMessage.user_id == user_id,
                    ChatMessage.sender == sender
                )
            )
            
            result = await session.execute(query)
            row = result.first()
            
            if not row:
                return None
            
            metrics = dict(row._mapping)
            
            # datetime 값 검증
            print(f"Retrieved datetime values - min: {metrics['min_datetime']}, max: {metrics['max_datetime']}")
            if not isinstance(metrics['min_datetime'], datetime) or not isinstance(metrics['max_datetime'], datetime):
                print(f"Warning: Invalid datetime types - min: {type(metrics['min_datetime'])}, max: {type(metrics['max_datetime'])}")
                return None
            
            return metrics

    async def extract_user_style(self, user: str, user_df: pd.DataFrame, chat_id: str, user_id: Optional[int] = None) -> Tuple[str, Dict, List[Dict]]:
        """개별 사용자의 언어 스타일 특성 추출 및 DB 저장"""
        morphology_results = []
        new_messages_df = user_df
        messages = user_df['preprocessed_message'].tolist()
        
        # 인증된 사용자인 경우 DB에서 기존 분석 결과 조회
        cached_metrics = None
        if user_id is not None:
            cached_metrics = await self._get_cached_metrics(chat_id, user_id, user)
            if cached_metrics:
                new_messages_df = user_df[
                    (user_df['datetime'] > cached_metrics['max_datetime']) |
                    (user_df['datetime'] < cached_metrics['min_datetime'])
                ]
                messages = new_messages_df['preprocessed_message'].tolist()
        
        # 공통 로직: 메시지 형태소 분석 수행
        total_morphemes = 0
        pos_totals = defaultdict(int)
        analyzed_messages = [] if user_id is not None else None
        
        # 각 메시지 개별 처리
        for message in messages:
            result = self.text_processor.analyze_morphology([message])
            morphology_results.append(result)
            
            # 전체 통계를 위한 집계
            total_morphemes += result['total_morphemes']
            for pos, count in result['pos_totals'].items():
                pos_totals[pos] += count
            
            # DB 저장용 메시지 데이터 생성
            if analyzed_messages is not None:
                idx = messages.index(message)
                row = new_messages_df.iloc[idx]
                analyzed_messages.append({
                    'chat_id': chat_id,
                    'user_id': user_id,
                    'datetime': row['datetime'],
                    'sender': user,
                    'message': row['message'],
                    'total_morphemes': result['total_morphemes'],  # 개별 메시지의 형태소 수
                    'substantives': result['pos_totals'].get('substantives', 0),  # 개별 메시지의 품사별 수
                    'predicates': result['pos_totals'].get('predicates', 0),
                    'endings': result['pos_totals'].get('endings', 0),
                    'modifiers': result['pos_totals'].get('modifiers', 0),
                    'expressions': result['pos_totals'].get('expressions', 0),
                    'question': 1 if '?' in row['message'] else 0,
                    'exclamation': 1 if '!' in row['message'] else 0,
                    'formal_ending': 1 if any(ending in row['message'] for ending in formal_endings) else 0
                })


        # 메시지 특성 벡터화 계산
        messages_array = np.array(messages)
        total_messages = len(messages)
        question_count = np.sum(['?' in msg for msg in messages])
        exclamation_count = np.sum(['!' in msg for msg in messages])
        
        # 형식적 종결어 패턴 매칭 벡터화
        formal_endings = ('습니다', '니다')
        formal_ending_count = np.sum([
            any(ending in msg for ending in formal_endings)
            for msg in messages
        ])

        # 저장된 메트릭이 있는 경우 누적
        if cached_metrics:
            total_morphemes += cached_metrics['total_morphemes']
            pos_totals['substantives'] += cached_metrics['substantives']
            pos_totals['predicates'] += cached_metrics['predicates']
            pos_totals['endings'] += cached_metrics['endings']
            pos_totals['modifiers'] += cached_metrics['modifiers']
            pos_totals['expressions'] += cached_metrics['expressions']
            total_messages += cached_metrics['message_count']
            question_count += cached_metrics['total_question']
            exclamation_count += cached_metrics['total_exclamation']
            formal_ending_count += cached_metrics['total_formal_ending']

            
        # 안전한 나눗셈을 위한 값
        total_morphemes_safe = max(total_morphemes, 1)
        # pos_totals를 pos_ratios로 벡터화 변환
        pos_ratios = {k: v/total_morphemes_safe for k, v in pos_totals.items()}
        
        # 스타일 특성 추출 - 벡터화 연산
        style_features = {
            'morphological_features': {
                'pos_ratios': pos_ratios,
                'total_morphemes': total_morphemes,
                'pos_totals': dict(pos_totals)
            },
            'syntactic_features': {
                'question_rate': question_count / max(total_messages, 1),
                'exclamation_rate': exclamation_count / max(total_messages, 1),
                'formal_ending_ratio': formal_ending_count / total_morphemes_safe,
                'avg_sentence_length': np.mean([len(result['pos_results']) for result in morphology_results]) if morphology_results else 0
            }
        }

        # 캐시 갱신
        if user_id is not None:
            metrics = {
                'min_datetime': new_messages_df['datetime'].min() if not new_messages_df.empty else cached_metrics['min_datetime'],
                'max_datetime': new_messages_df['datetime'].max() if not new_messages_df.empty else cached_metrics['max_datetime'],
                'message_count': total_messages,
                'total_morphemes': total_morphemes,
                'total_substantives': pos_totals['substantives'],
                'total_predicates': pos_totals['predicates'],
                'total_endings': pos_totals['endings'],
                'total_modifiers': pos_totals['modifiers'],
                'total_expressions': pos_totals['expressions'],
                'total_question': question_count,
                'total_exclamation': exclamation_count,
                'total_formal_ending': formal_ending_count,
            }
            await self._update_cache(chat_id, user_id, user, metrics)
        
        return user, style_features, analyzed_messages

    def _calculate_style_similarities(self, user_styles: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """사용자 간의 스타일 유사도 계산"""
        similarities = {}
        users = list(user_styles.keys())
        
        for i, user1 in enumerate(users):
            similarities[user1] = {}
            for user2 in users[i+1:]:
                # 각 특성 그룹별로 유사도 계산
                feature_similarities = {}
                for feature_type in ['morphological_features', 'syntactic_features']:
                    if feature_type == 'morphological_features':
                        # pos_ratios만 비교
                        vec1 = user_styles[user1][feature_type]['pos_ratios']
                        vec2 = user_styles[user2][feature_type]['pos_ratios']
                    else:
                        vec1 = user_styles[user1][feature_type]
                        vec2 = user_styles[user2][feature_type]
                    
                    similarity = self.text_processor.calculate_vector_similarity(vec1, vec2)
                    feature_similarities[feature_type] = similarity
                
                # 전체 유사도 계산 (가중 평균)
                weights = {'morphological_features': 0.6, 'syntactic_features': 0.4}
                total_similarity = sum(
                    similarity * weights[feature_type]
                    for feature_type, similarity in feature_similarities.items()
                )
                
                similarities[user1][user2] = total_similarity
                if user2 not in similarities:
                    similarities[user2] = {}
                similarities[user2][user1] = total_similarity
        
        return similarities

    async def _analyze_response_times(self, df: pd.DataFrame, participants: List[str]) -> Dict[str, Dict[str, float]]:
        """사용자별 응답 시간 분석"""

        
        # 사용자별 응답 시간 계산
        user_response_times = {}
        
        for user in participants:
            response_times = []
            user_messages = df[df['sender'] == user]
            
            for idx, row in user_messages.iterrows():
                # 현재 메시지 이전의 마지막 메시지 찾기
                prev_messages = df[
                    (df['datetime'] < row['datetime']) & 
                    (df['sender'] != user)
                ]
                
                if not prev_messages.empty:
                    last_message = prev_messages.iloc[-1]
                    time_diff = (row['datetime'] - last_message['datetime']).total_seconds()
                    response_times.append(time_diff)
            
            if response_times:
                # 큰 값의 이상치만 제거 (작은 값은 유지)
                adjusted_times = adjust_outliers(response_times, remove_lower=False)
                avg_response_time = float(np.mean(adjusted_times))
                
                user_response_times[user] = {
                    'average_response_time': avg_response_time,
                    'raw_response_count': len(response_times)
                }
        
        return user_response_times

    async def _update_cache(self, chat_id: str, user_id: int, sender: str, metrics: Dict):
        """분석 결과 캐시 갱신"""
        cache_key = self.cache_manager.generate_cache_key('interaction_stats', 
                                                         chat_id=chat_id, 
                                                         user_id=user_id, 
                                                         sender=sender)
        self.cache_manager.set('interaction_stats', cache_key, metrics)

    async def _store_analyzed_messages(self, messages: List[Dict]):
        if not messages:
            return
        await self.db_manager.store_chat_messages(messages)