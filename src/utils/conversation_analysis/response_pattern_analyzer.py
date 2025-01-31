from collections import defaultdict
from typing import Dict, List
import numpy as np
from .base_analyzer import ThreadMetricAnalyzer
from datetime import datetime
import asyncio

class ResponsePatternAnalyzer(ThreadMetricAnalyzer):
    def _initialize(self) -> None:
        self.response_times = []
        self.response_patterns = defaultdict(list)
    
    def on_first_message(self, sender: str, message: str, datetime: datetime) -> None:
        pass  # 첫 메시지는 응답 시간 계산 불가
    
    def on_speaker_change(self, prev_sender: str, curr_sender: str, 
                         message: str, time_diff: float) -> None:
        transition_key = f"{prev_sender}->{curr_sender}"
        self.response_times.append(time_diff)
        self.response_patterns[transition_key].append(time_diff)
    
    def on_continuous_message(self, sender: str, message: str) -> None:
        pass  # 연속 발화는 응답 시간 분석에서 제외
    
    def get_metrics(self) -> Dict:
        return {
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0,
            'response_time_by_pair': {
                k: {
                    'avg': np.mean(v),
                    'count': len(v)
                }
                for k, v in self.response_patterns.items()
            }
        } 

    async def merge_two_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict:
        """두 개의 메트릭을 병합"""
        # 응답 시간 병합을 위한 가중치(메시지 수) 계산
        count1 = sum(pattern['count'] for pattern in metrics1.get('response_time_by_pair', {}).values())
        count2 = sum(pattern['count'] for pattern in metrics2.get('response_time_by_pair', {}).values())
        total_count = count1 + count2

        if total_count == 0:
            return {
                'avg_response_time': 0,
                'response_time_by_pair': {}
            }

        # 전체 평균 응답 시간 계산 (가중 평균)
        avg_response_time = (
            metrics1.get('avg_response_time', 0) * count1 +
            metrics2.get('avg_response_time', 0) * count2
        ) / total_count

        # 화자 쌍별 응답 시간 병합
        merged_pairs = defaultdict(lambda: {'avg': 0, 'count': 0})
        
        for metrics in [metrics1, metrics2]:
            for pair, data in metrics.get('response_time_by_pair', {}).items():
                merged_pairs[pair]['avg'] = (
                    (merged_pairs[pair]['avg'] * merged_pairs[pair]['count'] +
                     data['avg'] * data['count']) /
                    (merged_pairs[pair]['count'] + data['count'])
                )
                merged_pairs[pair]['count'] += data['count']

        return {
            'avg_response_time': avg_response_time,
            'response_time_by_pair': dict(merged_pairs)
        }

    async def merge_metrics(self, metrics_list: List[Dict]) -> Dict:
        """여러 메트릭을 비동기적으로 병합"""
        if not metrics_list:
            return {
                'avg_response_time': 0,
                'response_time_by_pair': {}
            }
        
        if len(metrics_list) == 1:
            return metrics_list[0]

        async def merge_pair(pair: List[Dict]) -> Dict:
            if len(pair) == 1:
                return pair[0]
            return await self.merge_two_metrics(pair[0], pair[1])

        # 메트릭들을 두 개씩 병합
        while len(metrics_list) > 1:
            pairs = [metrics_list[i:i+2] for i in range(0, len(metrics_list), 2)]
            metrics_list = await asyncio.gather(*[merge_pair(pair) for pair in pairs])

        return metrics_list[0]

    