from collections import Counter, defaultdict
from typing import Dict, List
import numpy as np
from .base_analyzer import ThreadMetricAnalyzer
from datetime import datetime
import asyncio

class TurnTakingAnalyzer(ThreadMetricAnalyzer):
    def _initialize(self) -> None:
        self.turn_transitions = []
        # self.consecutive_turns = defaultdict(int)
        self.turn_lengths = defaultdict(list)
        self.turn_content_lengths = defaultdict(list)
        self.current_turn_length = 0
        self.current_turn_content_length = 0
    
    def on_first_message(self, sender: str, message: str, datetime: datetime) -> None:
        self.current_turn_length = 1
        self.current_turn_content_length = len(message)
    
    def on_speaker_change(self, prev_sender: str, curr_sender: str, 
                         message: str, time_diff: float) -> None:
        # 이전 턴 정보 저장
        self.turn_lengths[prev_sender].append(self.current_turn_length)
        self.turn_content_lengths[prev_sender].append(self.current_turn_content_length)
        
        # 턴 전환 기록
        transition = f"{prev_sender}->{curr_sender}"
        self.turn_transitions.append(transition)
        
        # 새 턴 시작
        self.current_turn_length = 1
        self.current_turn_content_length = len(message)
    
    def on_continuous_message(self, sender: str, message: str) -> None:
        # self.consecutive_turns[sender] += 1
        self.current_turn_length += 1
        self.current_turn_content_length += len(message)
    
    def get_metrics(self) -> Dict:
        return {
            'turn_transition_counts': dict(Counter(self.turn_transitions)),
            # 'avg_consecutive_turns': {
            #     k: v/sum(self.consecutive_turns.values())
            #     for k, v in self.consecutive_turns.items()
            # },
            'turn_metrics': {
                sender: {
                    'avg_messages_per_turn': np.mean(lengths) if lengths else 0,
                    'median_content_length_per_turn': np.median(self.turn_content_lengths[sender]) if self.turn_content_lengths[sender] else 0,
                    'total_turns': len(lengths)
                }
                for sender, lengths in self.turn_lengths.items()
            }
        } 

    async def merge_two_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict:
        """두 개의 턴테이킹 메트릭을 병합"""
        # 턴 전환 횟수 병합
        merged_transitions = defaultdict(int)
        for transition, count in metrics1.get('turn_transition_counts', {}).items():
            merged_transitions[transition] += count
        for transition, count in metrics2.get('turn_transition_counts', {}).items():
            merged_transitions[transition] += count

        # 턴 메트릭 병합
        merged_turn_metrics = {}
        all_senders = set(metrics1.get('turn_metrics', {}).keys()) | set(metrics2.get('turn_metrics', {}).keys())
        
        for sender in all_senders:
            metrics1_sender = metrics1.get('turn_metrics', {}).get(sender, {})
            metrics2_sender = metrics2.get('turn_metrics', {}).get(sender, {})
            
            total_turns1 = metrics1_sender.get('total_turns', 0)
            total_turns2 = metrics2_sender.get('total_turns', 0)
            total_turns = total_turns1 + total_turns2
            
            if total_turns == 0:
                continue
                
            # 가중 평균 계산
            merged_turn_metrics[sender] = {
                'avg_messages_per_turn': (
                    metrics1_sender.get('avg_messages_per_turn', 0) * total_turns1 +
                    metrics2_sender.get('avg_messages_per_turn', 0) * total_turns2
                ) / total_turns,
                'median_content_length_per_turn': (
                    metrics1_sender.get('median_content_length_per_turn', 0) * total_turns1 +
                    metrics2_sender.get('median_content_length_per_turn', 0) * total_turns2
                ) / total_turns,
                'total_turns': total_turns
            }

        return {
            'turn_transition_counts': dict(merged_transitions),
            'turn_metrics': merged_turn_metrics
        }

    async def merge_metrics(self, metrics_list: List[Dict]) -> Dict:
        """여러 턴테이킹 메트릭을 비동기적으로 병합"""
        if not metrics_list:
            return {
                'turn_transition_counts': {},
                'turn_metrics': {}
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

    