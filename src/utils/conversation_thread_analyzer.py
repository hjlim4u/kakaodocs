from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import networkx as nx
from scipy import stats
from .chat_utils import get_chat_id_from_df
from datetime import datetime
from .cache_manager import CacheManager
import asyncio

class ConversationThreadAnalyzer:
    def __init__(self):
        self.cache_manager = CacheManager()

    async def analyze_threads(self, df: pd.DataFrame, chat_id: str, thread_results: Dict) -> List[Dict]:
        """여러 대화 스레드를 비동기적으로 분석"""
        threads = thread_results['threads']
        
        # 각 유효 구간에 대한 분석 태스크 생성
        analysis_tasks = []
        for start_time, end_time in threads:
            task = asyncio.create_task(
                self.analyze_thread(df, start_time, end_time, chat_id)
            )
            analysis_tasks.append({
                'period': {
                    'start': start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'end': end_time.strftime("%Y-%m-%d %H:%M:%S")
                },
                'task': task
            })
        
        # 모든 구간 분석 완료 대기 및 결과 수집
        segment_analyses = []
        for analysis_item in analysis_tasks:
            analysis_result = await analysis_item['task']
            if analysis_result:  # 유효한 분석 결과만 포함
                segment_analyses.append({
                    'period': analysis_item['period'],
                    'analysis': analysis_result
                })
        
        return {
            'segments': segment_analyses,
            'thread_stats': thread_results['thread_stats']
        }

    async def analyze_thread(self, df: pd.DataFrame, start_time: datetime, end_time: datetime, chat_id: str) -> Dict:
        """각 대화 스레드에 대한 상세 분석"""
        # 시간 범위로 데이터 필터링
        thread_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
        
        # 빈 스레드 체크
        if len(thread_df) == 0:
            return self._get_empty_analysis_result(chat_id)
        
        # 각 분석 작업을 비동기 태스크로 생성
        tasks = [
            asyncio.create_task(self._get_basic_stats_async(thread_df)),
            asyncio.create_task(self._analyze_turn_taking_async(thread_df)),
            asyncio.create_task(self._analyze_response_patterns_async(thread_df)),
            asyncio.create_task(self._analyze_conversation_dynamics_async(thread_df))
        ]
        
        # 모든 분석 태스크 완료 대기
        basic_stats, turn_taking, response_patterns, conversation_dynamics = await asyncio.gather(*tasks)
        
        analysis_result = {
            'chat_id': chat_id,
            'period': {
                'start': start_time.strftime("%Y-%m-%d %H:%M:%S"),
                'end': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                'duration_seconds': (end_time - start_time).total_seconds()
            },
            'basic_stats': basic_stats,
            'turn_taking': turn_taking,
            'response_patterns': response_patterns,
            'conversation_dynamics': conversation_dynamics
        }
        
        return analysis_result

    async def _get_basic_stats_async(self, df: pd.DataFrame) -> Dict:
        """기초 통계 분석 - 비동기 버전"""
        return await asyncio.to_thread(self._get_basic_stats, df)

    async def _analyze_turn_taking_async(self, df: pd.DataFrame) -> Dict:
        """턴테이킹 패턴 분석 - 비동기 버전"""
        return await asyncio.to_thread(self._analyze_turn_taking, df)

    async def _analyze_response_patterns_async(self, df: pd.DataFrame) -> Dict:
        """응답 패턴 분석 - 비동기 버전"""
        return await asyncio.to_thread(self._analyze_response_patterns, df)

    async def _analyze_conversation_dynamics_async(self, df: pd.DataFrame) -> Dict:
        """대화 역학 분석 - 비동기 버전"""
        return await asyncio.to_thread(self._analyze_conversation_dynamics, df)

    def _get_basic_stats(self, df: pd.DataFrame) -> Dict:
        """특정 대화 구간의 기초 통계 분석"""
        message_lengths = df['message'].str.len()
        std_length = message_lengths.std() if len(df) > 1 else 0.0
        
        # 해당 구간의 참여자만 추출
        participants = sorted(df['sender'].unique().tolist())
        
        return {
            'duration': (df['datetime'].max() - df['datetime'].min()).total_seconds(),
            'message_count': len(df),
            'participant_count': len(participants),
            'avg_message_length': message_lengths.mean(),
            'message_length_std': std_length,
            'participants': participants,
            'messages_per_participant': df['sender'].value_counts().to_dict()
        }
        
    def _analyze_turn_taking(self, df: pd.DataFrame) -> Dict:
        """턴테이킹 패턴 분석"""
        # 단일 메시지 스레드 처리
        if len(df) <= 1:
            sender = df.iloc[0]['sender'] if len(df) == 1 else None
            return {
                'turn_transition_counts': {},
                'avg_consecutive_turns': {sender: 1.0} if sender else {},
                'avg_turn_length': {sender: 1.0} if sender else {},
                'turn_length_std': {sender: 0.0} if sender else {}
            }
        
        turn_transitions = []
        consecutive_turns = defaultdict(int)
        turn_lengths = defaultdict(list)
        
        prev_sender = None
        current_turn_length = 0
        
        for sender in df['sender']:
            if prev_sender:
                if sender != prev_sender:
                    # 튜플 대신 문자열로 저장
                    transition_key = f"{prev_sender}->{sender}"
                    turn_transitions.append(transition_key)
                    turn_lengths[prev_sender].append(current_turn_length)
                    current_turn_length = 1
                else:
                    consecutive_turns[sender] += 1
                    current_turn_length += 1
            prev_sender = sender
            
        # 마지막 턴 처리
        if prev_sender:
            turn_lengths[prev_sender].append(current_turn_length)
            
        return {
            'turn_transition_counts': dict(Counter(turn_transitions)),
            'avg_consecutive_turns': {k: v/len(df) for k, v in consecutive_turns.items()},
            'avg_turn_length': {k: np.mean(v) if v else 0 for k, v in turn_lengths.items()},
            'turn_length_std': {k: np.std(v) if len(v) > 1 else 0 for k, v in turn_lengths.items()}
        }
        
    def _analyze_response_patterns(self, df: pd.DataFrame) -> Dict:
        """응답 패턴 분석"""
        # 단일 메시지 또는 단일 사용자 구간 처리
        if len(df) <= 1 or df['sender'].nunique() == 1:
            return {
                'median_response_time': 0,
                'response_time_std': 0,
                'response_time_by_pair': {}
            }
        
        response_times = []
        response_patterns = defaultdict(list)
        
        for i in range(1, len(df)):
            curr_msg = df.iloc[i]
            prev_msg = df.iloc[i-1]
            
            if curr_msg['sender'] != prev_msg['sender']:
                time_diff = (curr_msg['datetime'] - prev_msg['datetime']).total_seconds()
                response_times.append(time_diff)
                pair_key = f"{prev_msg['sender']}->{curr_msg['sender']}"
                response_patterns[pair_key].append(time_diff)
        
        return {
            'median_response_time': np.median(response_times),
            'response_time_std': np.std(response_times) if len(response_times) > 1 else 0,
            'response_time_by_pair': {
                k: {
                    'median': np.median(v),
                    'std': np.std(v) if len(v) > 1 else 0,
                    'count': len(v)
                }
                for k, v in response_patterns.items()
            }
        }
        
    def _analyze_conversation_dynamics(self, df: pd.DataFrame) -> Dict:
        """대화 역학 분석"""
        if len(df) <= 1 or df['sender'].nunique() == 1:
            activity_pattern = {df.iloc[0]['datetime'].minute: 1} if len(df) == 1 else {}
            return {
                'activity_pattern': activity_pattern,
                'participation_inequality': 0,
                'flow_centrality': {
                    'in_degree': {},
                    'out_degree': {},
                    'betweenness': {}
                },
                'conversation_density': 0
            }
        
        # 필요한 컬럼만 새로운 Series로 생성
        minute_series = df['datetime'].dt.minute
        activity_by_minute = minute_series.value_counts()
        
        # 참여 불균형 측정
        participation_counts = df['sender'].value_counts().values
        participation_gini = self._calculate_gini(participation_counts) if len(participation_counts) > 0 else 0
        
        # 대화 흐름 그래프
        G = nx.DiGraph()
        for i in range(1, len(df)):
            prev_sender = df.iloc[i-1]['sender']
            curr_sender = df.iloc[i]['sender']
            if G.has_edge(prev_sender, curr_sender):
                G[prev_sender][curr_sender]['weight'] += 1
            else:
                G.add_edge(prev_sender, curr_sender, weight=1)
                
        # 빈 그래프 처리
        if len(G) == 0:
            return {
                'activity_pattern': activity_by_minute.to_dict(),
                'participation_inequality': 0,
                'flow_centrality': {
                    'in_degree': {},
                    'out_degree': {},
                    'betweenness': {}
                },
                'conversation_density': 0
            }
                
        # 중심성 측정값들을 문자열 키를 가진 딕셔너리로 변환
        centrality_measures = {
            'in_degree': {str(k): v for k, v in nx.in_degree_centrality(G).items()},
            'out_degree': {str(k): v for k, v in nx.out_degree_centrality(G).items()},
            'betweenness': {str(k): v for k, v in nx.betweenness_centrality(G).items()}
        }
        
        return {
            'activity_pattern': activity_by_minute.to_dict(),
            'participation_inequality': participation_gini,
            'flow_centrality': centrality_measures,
            'conversation_density': nx.density(G)
        }
        
    def _calculate_gini(self, values: np.ndarray) -> float:
        """지니 계수 계산 (참여 불균형 측정)"""
        if len(values) == 0:
            return 0
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * sorted_values).sum() / (n * sorted_values.sum()) if sorted_values.sum() > 0 else 0 

    def _get_empty_analysis_result(self, chat_id: str) -> Dict:
        """빈 분석 결과 반환"""
        return {
            'chat_id': chat_id,
            'period': {
                'start': '',
                'end': '',
                'duration_seconds': 0
            },
            'basic_stats': {
                'duration': 0,
                'message_count': 0,
                'participant_count': 0,
                'avg_message_length': 0,
                'message_length_std': 0,
                'participants': [],
                'messages_per_participant': {}
            },
            'turn_taking': {
                'turn_transition_counts': {},
                'avg_consecutive_turns': {},
                'avg_turn_length': {},
                'turn_length_std': {}
            },
            'response_patterns': {
                'median_response_time': 0,
                'response_time_std': 0,
                'response_time_by_pair': {}
            },
            'conversation_dynamics': {
                'activity_pattern': {},
                'participation_inequality': 0,
                'flow_centrality': {
                    'in_degree': {},
                    'out_degree': {},
                    'betweenness': {}
                },
                'conversation_density': 0
            }
        } 