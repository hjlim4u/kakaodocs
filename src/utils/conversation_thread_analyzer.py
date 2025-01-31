from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import networkx as nx
from scipy import stats
from .chat_utils import get_chat_id_from_df
from datetime import datetime
from .cache_manager import CacheManager
import asyncio
from .conversation_analysis import (
    TurnTakingAnalyzer,
    ResponsePatternAnalyzer,
    ConversationDynamicsAnalyzer,
)
from .conversation_analysis.base_analyzer import ThreadMetricAnalyzer

class ConversationThreadAnalyzer:
    def __init__(self, analyzers: Optional[List[ThreadMetricAnalyzer]] = None):
        self.cache_manager = CacheManager()
        self._analyzers = analyzers or self._create_default_analyzers()

    def _create_default_analyzers(self) -> List[ThreadMetricAnalyzer]:
        """기본 분석기 생성"""
        analyzers = [
            TurnTakingAnalyzer(),
            ResponsePatternAnalyzer(),
            ConversationDynamicsAnalyzer()
        ]
        
        # 각 분석기에 병합 메서드 추가
        for analyzer in analyzers:
            if not hasattr(analyzer, 'merge_metrics'):
                setattr(analyzer, 'merge_metrics', self._default_merge_metrics)
        
        return analyzers



    def register_analyzer(self, analyzer: ThreadMetricAnalyzer):
        """새로운 분석기 동적 추가"""
        self._analyzers.append(analyzer)

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

    def _notify_analyzers(self, event: str, *args):
        """분석기들에 대한 이벤트 전파
        
        Args:
            event: 이벤트 유형 ('first_message', 'speaker_change', 'continuous_message')
            args: 이벤트 핸들러에 전달할 인자들
        """
        handler_name = f'on_{event}'
        for analyzer in self._analyzers:
            handler = getattr(analyzer, handler_name)
            handler(*args)

    async def analyze_thread(self, thread_df: pd.DataFrame, chat_id: str, start_time: datetime = None, end_time: datetime = None) -> Dict:
        """각 대화 스레드에 대한 상세 분석"""
        if start_time and end_time:
            thread_df = thread_df[(thread_df['datetime'] >= start_time) & (thread_df['datetime'] <= end_time)]
        
        if len(thread_df) == 0:
            return self._get_empty_analysis_result(chat_id)
        
        # 모든 분석기 초기화
        for analyzer in self._analyzers:
            analyzer.initialize()
        
        prev_sender = None
        prev_datetime = None
        
        for idx, row in thread_df.iterrows():
            curr_sender = row['sender']
            curr_message = row['message']
            curr_datetime = row['datetime']
            
            # 이벤트 발생 및 모든 분석기에 전파
            if prev_sender is None:
                self._notify_analyzers('first_message', curr_sender, curr_message, curr_datetime)
            elif curr_sender != prev_sender:
                time_diff = (curr_datetime - prev_datetime).total_seconds()
                self._notify_analyzers('speaker_change', prev_sender, curr_sender, curr_message, time_diff)
            else:
                self._notify_analyzers('continuous_message', curr_sender, curr_message)
            
            prev_sender = curr_sender
            prev_datetime = curr_datetime
        
        # 분석 결과 수집
        return {
            analyzer.__class__.__name__.lower().replace('analyzer', ''): analyzer.get_metrics()
            for analyzer in self._analyzers
        }

    def _get_empty_analysis_result(self, chat_id: str) -> Dict:
        """빈 분석 결과 반환"""
        return {
            'chat_id': chat_id,
            'period': {
                'start': '',
                'end': '',
                'duration_seconds': 0
            },
            'turn_taking': {
                'turn_transition_counts': {},
                'avg_consecutive_turns': {},
                'turn_metrics': {}
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

    async def merge_thread_analyses(self, analyses: List[Dict]) -> Dict:
        """여러 스레드 분석 결과를 병합"""
        if not analyses:
            return self._get_empty_analysis_result('')
        
        merged_metrics = {}
        
        # 각 분석기별로 결과 병합
        for analyzer in self._analyzers:
            analyzer_name = analyzer.__class__.__name__.lower().replace('analyzer', '')
            metrics_list = [
                analysis.get(analyzer_name, {})
                for analysis in analyses
            ]
            
            # 각 분석기의 병합 메서드 호출
            merged_metrics[analyzer_name] = await analyzer.merge_metrics(metrics_list)
        
        # 기간 정보 업데이트
        start_times = [
            analysis.get('period', {}).get('start', None)
            for analysis in analyses
            if analysis.get('period', {}).get('start')
        ]
        end_times = [
            analysis.get('period', {}).get('end', None)
            for analysis in analyses
            if analysis.get('period', {}).get('end')
        ]
        
        if start_times and end_times:
            merged_metrics['period'] = {
                'start': min(start_times),
                'end': max(end_times),
                'duration_seconds': (max(end_times) - min(start_times)).total_seconds()
            }
        
        return merged_metrics 