import numpy as np
from typing import List, Dict
import pandas as pd
from collections import defaultdict

class PatternAnalyzer:
    async def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """대화 패턴 분석"""
        return {
            # 'response_times': self._analyze_response_times(df),  # ConversationThreadAnalyzer에서 처리
            'daily_patterns': self._analyze_daily_patterns(df),
            'message_patterns': self._analyze_message_patterns(df)
        }
        
    # def _analyze_response_times(self, df: pd.DataFrame) -> Dict:
    #     """응답 시간 패턴 분석"""
    #     response_times = []
    #     response_times_by_pair = {}
    #     
    #     for i in range(1, len(df)):
    #         curr_msg = df.iloc[i]
    #         prev_msg = df.iloc[i-1]
    #         
    #         if curr_msg['sender'] != prev_msg['sender']:
    #             time_diff = (curr_msg['datetime'] - prev_msg['datetime']).total_seconds()
    #             response_times.append(time_diff)
    #             
    #             pair_key = f"{prev_msg['sender']} -> {curr_msg['sender']}"
    #             if pair_key not in response_times_by_pair:
    #                 response_times_by_pair[pair_key] = []
    #             response_times_by_pair[pair_key].append(time_diff)
    #     
    #     # 전체 응답 시간 통계
    #     avg_response_time = np.mean(response_times) if response_times else 0
    #     median_response_time = np.median(response_times) if response_times else 0
    #     
    #     # 사용자 쌍별 응답 시간 통계
    #     pair_stats = {}
    #     for pair, times in response_times_by_pair.items():
    #         pair_stats[pair] = {
    #             'avg_time': np.mean(times),
    #             'median_time': np.median(times),
    #             'min_time': np.min(times),
    #             'max_time': np.max(times),
    #             'count': len(times)
    #         }
    #             
    #     return {
    #         'avg_response_time': avg_response_time,
    #         'median_response_time': median_response_time,
    #         'response_count': len(response_times),
    #         'pair_stats': pair_stats
    #     }
    
    def _analyze_daily_patterns(self, df: pd.DataFrame) -> Dict:
        """일별 패턴 분석"""
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        
        hourly_counts = df.groupby('hour').size().to_dict()
        weekday_counts = df.groupby('weekday').size().to_dict()
        
        return {
            'hourly_activity': hourly_counts,
            'weekday_activity': weekday_counts,
            'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0],
            'peak_weekday': max(weekday_counts.items(), key=lambda x: x[1])[0]
        }
    
    def _analyze_message_patterns(self, df: pd.DataFrame) -> Dict:
        """메시지 패턴 분석"""
        message_lengths = df['message'].str.len()
        
        return {
            'avg_length': message_lengths.mean(),
            'median_length': message_lengths.median(),
            'max_length': message_lengths.max(),
            'min_length': message_lengths.min(),
            'length_std': message_lengths.std()
        } 