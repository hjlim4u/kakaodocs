import asyncio
from typing import List, Tuple, Dict
import numpy as np
from scipy import stats
from datetime import datetime
import pandas as pd
from .text_processor import TextProcessor
from .cache_manager import CacheManager
from .db_manager import DatabaseManager
from sqlalchemy import select, and_
from ..database.models import ChatMessage
from bisect import bisect_right

class ThreadIdentifier:
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        self.cache_manager = CacheManager()
        self.DEFAULT_CLEAR_CUT = 14400  # 4시간
        self.MIN_AMBIGUOUS = 3600      # 1시간
        self.CONTEXT_TIME_THRESHOLD = 600  # 10분 (연속 메시지 수집 시간 임계값)

    async def identify_threads(self, df: pd.DataFrame, chat_id: str = None, user_id: int = None, db_manager: DatabaseManager = None) -> Dict:
        """대화 스레드를 식별"""

        last_analysis_period = None
        if user_id is not None and db_manager is not None:
            last_analysis_period = await self.identify_threads_with_stored(df, chat_id, user_id, db_manager)
        
        # 스레드 식별 (시간 기반)
        thread_periods = await self._identify_threads(df)
        
        if user_id is not None:
            if sorted(last_analysis_period.values()) not in thread_periods:
                df = df[
                    (df['datetime'] < last_analysis_period['start']) |
                    (df['datetime'] > last_analysis_period['end'])
                ]
            else:
                thread_periods.remove(sorted(last_analysis_period.values()))
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
        
        # self.cache_manager.set('thread', cache_key, result)
        return result

    def _is_valid_thread(self, thread_df: pd.DataFrame) -> bool:
        """유효한 대화 스레드인지 검증"""
        # 최소 2개 이상의 메시지
        # if len(thread_df) < 2:
        #     return False
        
        # 2명 이상의 참여자
        unique_senders = thread_df['sender'].unique()
        if len(unique_senders) < 2:
            return False
        
        return True

    # def _analyze_time_gaps(self, df: pd.DataFrame) -> Tuple[float, float, List[int]]:
    #     """시간 간격 분석을 통한 명확한 대화 구분점 식별"""
    #     # 기본 임계값 설정 (단위: 초)
    #     DEFAULT_CLEAR_CUT = 14400   # 4시간
    #     # DEFAULT_AMBIGUOUS = 3600   # 1시간
    #     # MIN_CLEAR_CUT = 3600      # 1시간
    #     MIN_AMBIGUOUS = 3600      # 1시간
        
    #     time_diffs = []
    #     indices = []
        
    #     for i in range(1, len(df)):
    #         curr_row = df.iloc[i]
    #         prev_row = df.iloc[i-1]
    #         time_diff = (curr_row['datetime'] - prev_row['datetime']).total_seconds()
            
    #         # 같은 발화자의 연속된 메시지인 경우
    #         if curr_row['sender'] == prev_row['sender']:
    #             # 긴 시간 간격(예: 1시간 이상)이 있는 경우는 새로운 대화의 시작으로 간주
    #             if time_diff >= MIN_AMBIGUOUS: 
    #                 time_diffs.append(time_diff)
    #                 indices.append(i)
    #             continue
            
    #         time_diffs.append(time_diff)
    #         indices.append(i)
        
    #     # 분석할 시간 간격이 없는 경우 처리
    #     if not time_diffs:
    #         return DEFAULT_CLEAR_CUT, DEFAULT_AMBIGUOUS, []
            
    #     time_diffs = np.array(time_diffs)
    #     median = np.median(time_diffs)
    #     mad = stats.median_abs_deviation(time_diffs)
    #     modified_z_scores = 0.6745 * (time_diffs - median) / mad
        
    #     # Modified Z-score로 계산된 임계값
    #     z_clear_cuts = modified_z_scores > 5.0
    #     z_clear_threshold = np.min(time_diffs[z_clear_cuts]) if np.any(z_clear_cuts) else np.percentile(time_diffs, 98)
    #     z_ambiguous_threshold = np.min(time_diffs[modified_z_scores > 3.5]) if np.any(modified_z_scores > 3.5) else np.percentile(time_diffs, 90)
        
    #     # 최종 임계값 결정 (기본값, Z-score, 최소값 고려)
    #     clear_cut_threshold = min(max(z_clear_threshold, MIN_CLEAR_CUT), DEFAULT_CLEAR_CUT)
    #     ambiguous_threshold = min(max(z_ambiguous_threshold, MIN_AMBIGUOUS), DEFAULT_AMBIGUOUS)
        
    #     # 명확한 구분점 식별
    #     clear_cut_indices = [idx for idx, diff in zip(indices, time_diffs) if diff >= clear_cut_threshold]
        
    #     # 임계값 출력
    #     print("\n=== 대화 구분 임계값 ===")
    #     print(f"데이터 수: {len(time_diffs)}개")
    #     print(f"명확한 구분점 임계값: {clear_cut_threshold:.1f}초 ({clear_cut_threshold/60:.1f}분)")
    #     print(f"애매한 구간 임계값: {ambiguous_threshold:.1f}초 ({ambiguous_threshold/60:.1f}분)")
    #     print(f"중앙값: {median:.1f}초 ({median/60:.1f}분)")
    #     print(f"MAD: {mad:.1f}초")
    #     print(f"Z-score 기반 임계값 - 명확: {z_clear_threshold:.1f}초, 애매: {z_ambiguous_threshold:.1f}초")
        
    #     return clear_cut_threshold, ambiguous_threshold, clear_cut_indices

    async def _analyze_message_similarity(self, messages: List[str]) -> float:
        """두 메시지 간의 의미적 유사도 분석"""
        embeddings = await self.text_processor.get_embeddings(messages)
        similarity = self.text_processor.calculate_vector_similarity(
            embeddings[0],
            embeddings[1]
        )

        return similarity

    def _collect_context_messages(self, df: pd.DataFrame, idx: int, time_threshold: float) -> Tuple[List[str], List[str]]:
        """이전/이후 연속된 메시지들을 수집"""
        before_messages = []
        after_messages = []
        
        # 이전 메시지들 수집
        curr_idx = idx - 1
        curr_sender = df.iloc[curr_idx]['sender']
        while curr_idx > 0:
            prev_idx = curr_idx - 1
            time_diff = (df.iloc[curr_idx]['datetime'] - df.iloc[prev_idx]['datetime']).total_seconds()
            if time_diff > time_threshold or df.iloc[prev_idx]['sender'] != curr_sender:
                break
            before_messages.insert(0, df.iloc[prev_idx]['message'])
            curr_idx = prev_idx
        before_messages.append(df.iloc[idx-1]['message'])
        
        # 이후 메시지들 수집
        curr_idx = idx
        curr_sender = df.iloc[curr_idx]['sender']
        while curr_idx < len(df) - 1:
            next_idx = curr_idx + 1
            time_diff = (df.iloc[next_idx]['datetime'] - df.iloc[curr_idx]['datetime']).total_seconds()
            if time_diff > time_threshold or df.iloc[next_idx]['sender'] != curr_sender:
                break
            after_messages.append(df.iloc[next_idx]['message'])
            curr_idx = next_idx
        after_messages.insert(0, df.iloc[idx]['message'])
        
        return before_messages, after_messages

    async def _identify_threads(self, df: pd.DataFrame) -> List[List[datetime]]:
        """시간 간격과 의미적 유사도를 기반으로 대화 스레드 식별"""
        # 고정 임계값 설정 (단위: 초)
        DEFAULT_CLEAR_CUT = 14400  # 4시간
        MIN_AMBIGUOUS = 3600      # 1시간
        CONTEXT_TIME_THRESHOLD = 600  # 10분 (연속 메시지 수집 시간 임계값)
        
        # 명확한 구분점과 애매한 구간 분석을 위한 배열
        clear_cuts = []  # 명확한 구분점 (인덱스)
        ambiguous_tasks = []  # (인덱스, 유사도 분석 태스크) 튜플 리스트
        
        print("\n=== 대화 구분점 분석 ===")
        for i in range(1, len(df)):
            time_diff = (df.iloc[i]['datetime'] - df.iloc[i-1]['datetime']).total_seconds()
            
            # 명확한 구분점
            if time_diff >= DEFAULT_CLEAR_CUT:
                clear_cuts.append(i)
                continue
                
            # 애매한 구간 분석을 위한 태스크 생성
            if time_diff >= MIN_AMBIGUOUS:
                before_msgs, after_msgs = self._collect_context_messages(df, i, CONTEXT_TIME_THRESHOLD)
                before_context = " ".join(before_msgs)
                after_context = " ".join(after_msgs)
                similarity_task = asyncio.create_task(self._analyze_message_similarity([before_context, after_context]))
                ambiguous_tasks.append((i, similarity_task, before_msgs, after_msgs))

        # 모든 유사도 분석 태스크 완료 대기
        all_cuts = clear_cuts.copy()
        if ambiguous_tasks:
            print("\n애매한 구간 분석 중...")
            for idx, task, before_msgs, after_msgs in ambiguous_tasks:
                similarity = await task
                print(f"\n시간 간격: {(df.iloc[idx]['datetime'] - df.iloc[idx-1]['datetime']).total_seconds():.1f}초")
                print(f"이전 문맥 ({len(before_msgs)}개 메시지): {before_msgs}")
                print(f"이후 문맥 ({len(after_msgs)}개 메시지): {after_msgs}")
                print(f"의미적 유사도: {similarity:.3f}")
                
                if similarity < 0.5:  # 유사도 임계값
                    all_cuts.append(idx)

        # 구분점 정렬
        all_cuts.sort()
        
        # 대화 스레드 생성
        threads = []
        start_idx = 0
        
        for cut in all_cuts:
            if start_idx < cut:  # 빈 스레드 제외
                threads.append([
                    df.iloc[start_idx]['datetime'],  # 시작 시간
                    df.iloc[cut - 1]['datetime']     # 끝 시간
                ])
            start_idx = cut
        
        # 마지막 스레드 추가
        if start_idx < len(df):
            threads.append([
                df.iloc[start_idx]['datetime'],      # 시작 시간
                df.iloc[len(df) - 1]['datetime']     # 끝 시간
            ])
        
        return threads

    def _generate_cache_key(self, chat_id: str, df: pd.DataFrame, suffix: str = "") -> str:
        """캐시 키 생성"""
        base_key = f"{df.iloc[0]['datetime']}_{df.iloc[-1]['datetime']}"
        return f"{chat_id}:{base_key}{suffix}"

        """이분 탐색으로 가장 가까운 시간의 인덱스와 시간 차이를 찾음"""
        times = df['datetime'].values
        left, right = 0, len(times) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if times[mid] == target_time:
                return mid, 0
            elif times[mid] < target_time:
                left = mid + 1
            else:
                right = mid - 1
        
        # 가장 가까운 인덱스 찾기
        candidates = []
        if right >= 0:
            candidates.append((right, abs((target_time - times[right]).total_seconds())))
        if left < len(times):
            candidates.append((left, abs((target_time - times[left]).total_seconds())))
            
        return min(candidates, key=lambda x: x[1])

    async def identify_threads_with_stored(self, df: pd.DataFrame, chat_id: str, user_id: int, db_manager: DatabaseManager) -> Dict:
        """DB에 저장된 구간을 고려하여 대화 스레드를 식별"""
        # DB에서 마지막 구간 분석 결과만 조회
        last_analysis = await db_manager.get_last_thread_analysis(chat_id, user_id)
        
        if not last_analysis:
            return df, last_analysis
        
        # 마지막 구간의 끝 시점이 삽입될 위치 찾기
        times = df['datetime'].values
        last_end_time = last_analysis['period']['end']
        idx = bisect_right(times, last_end_time)
        
        # 인접한 시간과의 차이 계산
        time_diff = float('inf')
        # if idx > 0:
        #     time_diff = min(time_diff, abs((last_end_time - times[idx-1]).total_seconds()))
        if idx < len(times):
            time_diff = min(time_diff, abs((times[idx] - last_end_time).total_seconds()))
        
        # 시간 차이가 임계값보다 작으면 DB에서 해당 구간의 메시지들 가져와서 병합
        if time_diff <= self.DEFAULT_CLEAR_CUT:
            stored_messages = await db_manager.get_messages_in_period(chat_id, user_id, last_analysis['period']['start'], last_analysis['period']['end'])
                
            stored_df = pd.DataFrame([{
                'datetime': msg.datetime,
                'sender': msg.sender,
                'message': msg.message
            } for msg in stored_messages])
            stored_df['preprocessed_message'] = stored_df['message'].apply(self.text_processor.preprocess_message)
            
            if not stored_df.empty:
                # concat 결과를 df에 직접 할당
                df = pd.concat([df, stored_df])
                df.sort_values('datetime', inplace=True)
                df.drop_duplicates(
                    subset=['datetime', 'sender', 'message'], 
                    inplace=True
                )
                
        return last_analysis['period']
