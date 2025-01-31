from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from typing import Dict, List
import numpy as np
import asyncio
from .text_processor import TextProcessor
from .chat_utils import get_chat_id_from_df
from .cache_manager import CacheManager

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "beomi/KcELECTRA-base-v2022",
            num_labels=2  # positive/negative 분류
        )
        self.model.eval()
        self.cache_manager = CacheManager()
        
    async def analyze_emotion(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits을 확률로 변환
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # 값 추출
            positive_score = float(probabilities[0][1])
            negative_score = float(probabilities[0][0])
            
            # neutral 점수 계산 (양극단에서 멀어진 정도)
            neutral_score = 1.0 - abs(positive_score - negative_score)
            
            # 모든 점수를 정규화
            total = positive_score + negative_score + neutral_score
            if total > 0:
                positive_score = positive_score / total
                negative_score = negative_score / total
                neutral_score = neutral_score / total
            else:
                positive_score = negative_score = neutral_score = 1/3
            
            return {
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score
            }
        
    async def  analyze_sentiments(self, df: pd.DataFrame, chat_id: str, thread_results: Dict) -> Dict:
        """감정 분석 수행"""
        threads = thread_results['threads']
        
        # 메시지 감정 분석 병렬 처리
        async def analyze_messages(messages):
            tasks = [self.analyze_emotion(msg) for msg in messages]
            return await asyncio.gather(*tasks)
        
        # # 참여자 목록 캐시 활용
        # participants = self.cache_manager.get('participants', chat_id)
        # if participants is None:
        #     participants = sorted(df['sender'].unique())
        #     self.cache_manager.set('participants', chat_id, participants)
        
        # # 사용자별 감정 분석 병렬 처리
        # user_tasks = []
        # for user in participants:
        #     user_messages = df[df['sender'] == user]['preprocessed_message']
        #     user_tasks.append(analyze_messages(user_messages))
        
        # user_emotions = await asyncio.gather(*user_tasks)

        # 사용자별 감정 분석
        # user_sentiments = {}
        # for user, emotions in zip(participants, user_emotions):
        #     user_sentiments[user] = {
        #         'positive': np.mean([e['positive'] for e in emotions]),
        #         'negative': np.mean([e['negative'] for e in emotions]),
        #         'neutral': np.mean([e['neutral'] for e in emotions]),
        #         'sentiment_std': np.std([e['positive'] - e['negative'] for e in emotions]),
        #         'message_count': len(emotions)
        #     }
        
        # 스레드별 감정 흐름 분석
        sentiment_flow = []
        thread_analysis_tasks = []
        
        for start_time, end_time in threads:
            # 해당 스레드의 메시지들 추출
            thread_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
            thread_messages = thread_df['preprocessed_message'].tolist()
            thread_analysis_tasks.append({
                'task': analyze_messages(thread_messages),
                'start_time': start_time,
                'end_time': end_time
            })
        
        # 모든 스레드 분석 완료 대기
        for analysis_item in thread_analysis_tasks:
            thread_emotions = await analysis_item['task']
            sentiment_flow.append({
                'start_time': analysis_item['start_time'],
                'end_time': analysis_item['end_time'],
                'avg_sentiment': {
                    'positive': np.mean([e['positive'] for e in thread_emotions]),
                    'negative': np.mean([e['negative'] for e in thread_emotions]),
                    'neutral': np.mean([e['neutral'] for e in thread_emotions])
                },
                # 'message_count': len(thread_emotions)
            })
        
        return {
            'chat_id': chat_id,
            # 'user_sentiments': user_sentiments,
            'sentiment_flow': sentiment_flow,
            'overall_sentiment': {
                'positive': np.mean([s['avg_sentiment']['positive'] for s in sentiment_flow]) if sentiment_flow else 0,
                'negative': np.mean([s['avg_sentiment']['negative'] for s in sentiment_flow]) if sentiment_flow else 0,
                'neutral': np.mean([s['avg_sentiment']['neutral'] for s in sentiment_flow]) if sentiment_flow else 0
            }
        } 

    async def analyze_sentiment(self, thread_df: pd.DataFrame, chat_id: str) -> Dict:
        """단일 스레드의 감정 분석 수행"""
        if len(thread_df) == 0:
            return {
                'chat_id': chat_id,
                'sentiment_flow': [],
                'overall_sentiment': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            }

        # 메시지 감정 분석 병렬 처리
        async def analyze_messages(messages):
            tasks = [self.analyze_emotion(msg) for msg in messages]
            return await asyncio.gather(*tasks)
        prev_sender = None
        user_messages = defaultdict(list)
        for _, row in thread_df[['preprocessed_message', 'sender']].iterrows():
            msg = row['preprocessed_message']
            sender = row['sender']
            print(msg, sender)
            if prev_sender != sender:
                user_messages[sender].append(msg)
                prev_sender = sender
                print(prev_sender)
            else:
                user_messages[sender][-1] += msg

        # 스레드의 메시지들 추출
        # thread_messages = thread_df['preprocessed_message'].tolist()
        users_emotions = defaultdict(list)
        for user, messages in user_messages.items():
            users_emotions[user] = await analyze_messages(messages)

        # # 감정 분석 수행
        # thread_emotions = await analyze_messages(thread_messages)
        
        # 전체 구간의 평균 감정
        all_emotions = []
        for emotions_list in users_emotions.values():
            all_emotions.extend(emotions_list)  # emotions_list는 리스트의 리스트

        avg_sentiment = {
            'positive': np.mean([e['positive'] for e in all_emotions]),
            'negative': np.mean([e['negative'] for e in all_emotions]),
            'neutral': np.mean([e['neutral'] for e in all_emotions])
        }

        return {
            'chat_id': chat_id,
            'start_time': thread_df['datetime'].min(),
            'end_time': thread_df['datetime'].max(),
            'users_emotions': {
                user: {
                    'positive': np.mean([e['positive'] for e in emotions]),
                    'negative': np.mean([e['negative'] for e in emotions]),
                    'neutral': np.mean([e['neutral'] for e in emotions])
                } for user, emotions in users_emotions.items()
            },
            'avg_sentiment': avg_sentiment,
            'message_counts': {  # 각 사용자의 메시지 수 저장
                user: len(emotions) for user, emotions in users_emotions.items()
            }
        } 

    async def merge_two_analyses(self, analysis1: Dict, analysis2: Dict) -> Dict:
        """두 개의 감정 분석 결과를 병합"""
        # 시간 흐름 데이터 병합 및 정렬
        sentiment_flow = sorted(
            analysis1['sentiment_flow'] + analysis2['sentiment_flow'],
            key=lambda x: x['start_time']
        )
        
        # 사용자별 감정과 메시지 수 병합
        merged_users_emotions = {}
        merged_message_counts = {}
        
        for user in set(analysis1.get('users_emotions', {}).keys()) | set(analysis2.get('users_emotions', {}).keys()):
            count1 = analysis1.get('message_counts', {}).get(user, 0)
            count2 = analysis2.get('message_counts', {}).get(user, 0)
            total_count = count1 + count2
            
            if total_count == 0:
                continue
                
            emotions1 = analysis1.get('users_emotions', {}).get(user, {'positive': 0, 'negative': 0, 'neutral': 0})
            emotions2 = analysis2.get('users_emotions', {}).get(user, {'positive': 0, 'negative': 0, 'neutral': 0})
            
            # 가중 평균 계산
            merged_users_emotions[user] = {
                sentiment_type: (
                    emotions1[sentiment_type] * count1 + 
                    emotions2[sentiment_type] * count2
                ) / total_count
                for sentiment_type in ['positive', 'negative', 'neutral']
            }
            merged_message_counts[user] = total_count
            
        # 전체 감정 계산 (메시지 수 기반 가중 평균)
        total_messages = sum(merged_message_counts.values())
        overall_sentiment = {
            sentiment_type: sum(
                emotions[sentiment_type] * merged_message_counts[user] 
                for user, emotions in merged_users_emotions.items()
            ) / total_messages if total_messages > 0 else 0
            for sentiment_type in ['positive', 'negative', 'neutral']
        }
        
        return {
            'chat_id': analysis1['chat_id'],
            'users_emotions': merged_users_emotions,
            'message_counts': merged_message_counts,
            'sentiment_flow': sentiment_flow,
            'overall_sentiment': overall_sentiment
        }

    async def merge_sentiment_analyses(self, analyses: List[Dict]) -> Dict:
        """여러 감정 분석 결과를 병합"""
        if not analyses:
            return {
                'chat_id': '',
                'sentiment_flow': [],
                'overall_sentiment': {
                    'positive': 0, 'negative': 0, 'neutral': 0
                },
                'users_emotions': {},
                'message_counts': {}
            }
        
        if len(analyses) == 1:
            return analyses[0]
            
        # 두 개씩 병합하는 작업을 비동기적으로 수행
        async def merge_pair(pair: List[Dict]) -> Dict:
            if len(pair) == 1:
                return pair[0]
            return await self.merge_two_analyses(pair[0], pair[1])
            
        while len(analyses) > 1:
            # 분석 결과들을 두 개씩 쌍으로 나누기
            pairs = [analyses[i:i+2] for i in range(0, len(analyses), 2)]
            # 각 쌍을 비동기적으로 병합
            analyses = await asyncio.gather(*[merge_pair(pair) for pair in pairs])
            
        return analyses[0]

