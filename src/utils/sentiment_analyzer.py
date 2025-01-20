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
        
    async def analyze_sentiments(self, df: pd.DataFrame, chat_id: str, thread_results: Dict) -> Dict:
        """감정 분석 수행"""
        threads = thread_results['threads']
        
        # 메시지 감정 분석 병렬 처리
        async def analyze_messages(messages):
            tasks = [self.analyze_emotion(msg) for msg in messages]
            return await asyncio.gather(*tasks)
        
        # 참여자 목록 캐시 활용
        participants = self.cache_manager.get('participants', chat_id)
        if participants is None:
            participants = sorted(df['sender'].unique())
            self.cache_manager.set('participants', chat_id, participants)
        
        # 사용자별 감정 분석 병렬 처리
        user_tasks = []
        for user in participants:
            user_messages = df[df['sender'] == user]['message']
            user_tasks.append(analyze_messages(user_messages))
        
        user_emotions = await asyncio.gather(*user_tasks)
        
        # 사용자별 감정 분석
        user_sentiments = {}
        for user, emotions in zip(participants, user_emotions):
            user_sentiments[user] = {
                'positive': np.mean([e['positive'] for e in emotions]),
                'negative': np.mean([e['negative'] for e in emotions]),
                'neutral': np.mean([e['neutral'] for e in emotions]),
                'sentiment_std': np.std([e['positive'] - e['negative'] for e in emotions]),
                'message_count': len(emotions)
            }
        
        # 스레드별 감정 흐름 분석
        sentiment_flow = []
        thread_analysis_tasks = []
        
        for start_time, end_time in threads:
            # 해당 스레드의 메시지들 추출
            thread_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
            thread_messages = thread_df['message'].tolist()
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
                'message_count': len(thread_emotions)
            })
        
        return {
            'chat_id': chat_id,
            'user_sentiments': user_sentiments,
            'sentiment_flow': sentiment_flow,
            'overall_sentiment': {
                'positive': np.mean([s['avg_sentiment']['positive'] for s in sentiment_flow]) if sentiment_flow else 0,
                'negative': np.mean([s['avg_sentiment']['negative'] for s in sentiment_flow]) if sentiment_flow else 0,
                'neutral': np.mean([s['avg_sentiment']['neutral'] for s in sentiment_flow]) if sentiment_flow else 0
            }
        } 