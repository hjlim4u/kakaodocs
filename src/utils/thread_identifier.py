import asyncio
from typing import List, Tuple, Dict
import numpy as np
from scipy import stats
from datetime import datetime
import pandas as pd
from .text_processor import TextProcessor
from .cache_manager import CacheManager

class ThreadIdentifier:
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        self.cache_manager = CacheManager()

    async def identify_threads(self, df: pd.DataFrame, chat_id: str) -> Dict:
        """대화 스레드를 식별"""
        cache_key = self._generate_cache_key(chat_id, df, ":threads")
        
        cached_result = self.cache_manager.get('thread', cache_key)
        if cached_result is not None:
            return cached_result
            
        # 스레드 식별 (시간 기반)
        thread_periods = await self._identify_threads(df)
        
        # 유효한 스레드만 필터링
        valid_threads = []
        for start_time, end_time in thread_periods:
            thread_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
            if self._is_valid_thread(thread_df):
                valid_threads.append([start_time, end_time])
        
        result = {
            'threads': valid_threads,
            'thread_stats': {
                'total_threads': len(valid_threads),
                'avg_thread_length': np.mean([
                    (end - start).total_seconds() 
                    for start, end in valid_threads
                ]) if valid_threads else 0,
                'max_thread_length': max([
                    (end - start).total_seconds() 
                    for start, end in valid_threads
                ]) if valid_threads else 0
            }
        }
        
        self.cache_manager.set('thread', cache_key, result)
        return result

    def _is_valid_thread(self, thread_df: pd.DataFrame) -> bool:
        """유효한 대화 스레드인지 검증"""
        # 최소 2개 이상의 메시지
        if len(thread_df) < 2:
            return False
        
        # 2명 이상의 참여자
        unique_senders = thread_df['sender'].unique()
        if len(unique_senders) < 2:
            return False
        
        # 실제 대화 여부 확인 (단순 연속 발화 제외)
        has_interaction = False
        prev_sender = thread_df.iloc[0]['sender']
        
        for _, row in thread_df.iloc[1:].iterrows():
            if row['sender'] != prev_sender:
                has_interaction = True
                break
            prev_sender = row['sender']
        
        return has_interaction

    def _analyze_time_gaps(self, df: pd.DataFrame) -> Tuple[float, float, List[int]]:
        """시간 간격 분석을 통한 명확한 대화 구분점 식별"""
        time_diffs = []
        indices = []
        
        for i in range(1, len(df)):
            curr_time = df.iloc[i]['datetime']
            prev_time = df.iloc[i-1]['datetime']
            diff = (curr_time - prev_time).total_seconds()
            time_diffs.append(diff)
            indices.append(i)
            
        time_diffs = np.array(time_diffs)
        
        # Modified Z-score method로 이상치 탐지
        median = np.median(time_diffs)
        mad = stats.median_abs_deviation(time_diffs)
        modified_z_scores = 0.6745 * (time_diffs - median) / mad
        
        # 명확한 구분점 (Z-score > 3.5)
        clear_cuts = modified_z_scores > 3.5
        clear_cut_threshold = np.min(time_diffs[clear_cuts]) if np.any(clear_cuts) else np.percentile(time_diffs, 95)
        
        # 애매한 구간 (2.5 < Z-score <= 3.5)
        ambiguous_threshold = np.min(time_diffs[modified_z_scores > 2.5]) if np.any(modified_z_scores > 2.5) else np.percentile(time_diffs, 75)
        
        clear_cut_indices = [idx for idx, is_cut in zip(indices, clear_cuts) if is_cut]
        
        return clear_cut_threshold, ambiguous_threshold, clear_cut_indices

    async def _analyze_message_similarity(self, messages: List[str]) -> float:
        """두 메시지 간의 의미적 유사도 분석"""
        embeddings = await self.text_processor.get_embeddings(messages)
        similarity = self.text_processor.calculate_semantic_similarity(
            embeddings[0],
            embeddings[1]
        )
        return similarity

    async def _identify_threads(self, df: pd.DataFrame) -> List[List[datetime]]:
        """시간 간격과 의미적 유사도를 기반으로 대화 스레드 식별"""
        # 1단계: 시간 간격 분석
        clear_cut_threshold, ambiguous_threshold, clear_cuts = self._analyze_time_gaps(df)
        
        # 애매한 구간에 대한 의미적 유사도 분석 태스크 생성
        similarity_tasks = []
        all_cuts = []
        clear_cuts_idx = 0
        
        for i in range(1, len(df)):
            time_diff = (df.iloc[i]['datetime'] - df.iloc[i-1]['datetime']).total_seconds()
            if ambiguous_threshold <= time_diff < clear_cut_threshold:
                messages = [df.iloc[i-1]['message'], df.iloc[i]['message']]
                task = asyncio.create_task(self._analyze_message_similarity(messages))
                similarity_tasks.append((i, task))
        
        # 의미적 유사도 분석 태스크 완료 대기 및 결과 처리
        for idx, task in similarity_tasks:
            while clear_cuts_idx < len(clear_cuts) and clear_cuts[clear_cuts_idx] < idx:
                all_cuts.append(clear_cuts[clear_cuts_idx])
                clear_cuts_idx += 1
                
            similarity = await task
            if similarity < 0.4:  # similarity_threshold
                all_cuts.append(idx)
        
        # 남은 clear_cuts 추가
        while clear_cuts_idx < len(clear_cuts):
            all_cuts.append(clear_cuts[clear_cuts_idx])
            clear_cuts_idx += 1
        
        # 대화 스레드 생성
        threads = []
        start_idx = 0
        
        for cut in all_cuts:
            if start_idx < cut:  # 빈 스레드 제외
                threads.append([
                    df.iloc[start_idx]['datetime'],  # 시작 시간
                    df.iloc[cut - 1]['datetime']    # 끝 시간
                ])
            start_idx = cut
        
        # 마지막 스레드 추가
        if start_idx < len(df):
            threads.append([
                df.iloc[start_idx]['datetime'],      # 시작 시간
                df.iloc[len(df) - 1]['datetime']    # 끝 시간
            ])
        
        return threads

    def _generate_cache_key(self, chat_id: str, df: pd.DataFrame, suffix: str = "") -> str:
        """캐시 키 생성"""
        base_key = f"{df.iloc[0]['datetime']}_{df.iloc[-1]['datetime']}"
        return f"{chat_id}:{base_key}{suffix}"
