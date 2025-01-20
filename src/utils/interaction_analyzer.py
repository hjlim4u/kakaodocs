import networkx as nx
import numpy as np
from typing import Dict
import pandas as pd
from .text_processor import TextProcessor
from .cache_manager import CacheManager

class InteractionAnalyzer:
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        self.cache_manager = CacheManager()

    async def analyze_interactions(self, df: pd.DataFrame, chat_id: str, thread_results: Dict) -> Dict:
        """
        대화 상호작용 분석 수행
        """
        threads = thread_results['threads']
        
        # 상호작용 그래프 생성
        G = nx.Graph()
        
        # 각 대화 스레드 내에서 상호작용 분석
        for thread in threads:
            start_time, end_time = thread
            thread_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
            
            prev_sender = None
            for _, msg in thread_df.iterrows():
                if prev_sender and prev_sender != msg['sender']:
                    if G.has_edge(prev_sender, msg['sender']):
                        G[prev_sender][msg['sender']]['weight'] += 1
                    else:
                        G.add_edge(prev_sender, msg['sender'], weight=1)
                prev_sender = msg['sender']
        
        # 전체 분석 결과 생성
        language_style_results = self._analyze_language_style(df, chat_id)
        return {
            'network': G,
            'centrality': {
                'degree': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G),
                'closeness': nx.closeness_centrality(G)
            },
            'density': nx.density(G),
            'thread_stats': thread_results['thread_stats'],
            'threads': threads,
            'language_style_analysis': language_style_results
        }

    def _analyze_language_style(self, df: pd.DataFrame, chat_id: str = None) -> Dict:
        """대화 참여자들의 언어 스타일 분석"""
        try:
            cache_key = self.cache_manager.generate_cache_key(
                'pos',
                chat_id=chat_id,
                suffix='language_style'
            )
            if cache_key:
                cached_result = self.cache_manager.get('pos', cache_key)
                if cached_result is not None:
                    return cached_result
        except ValueError:
            cache_key = None
        
        user_styles = {}
        style_similarities = {}
        
        # 참여자 목록을 캐시에서 조회
        participants = self.cache_manager.get('participants', chat_id)
        if participants is None:
            # 캐시에 없는 경우 계산 후 저장
            participants = sorted(df['sender'].unique().tolist())
            self.cache_manager.set('participants', chat_id, participants)
        
        # 사용자별 언어 스타일 특성 추출
        for sender in participants:
            user_messages = df[df['sender'] == sender]['message'].tolist()
            user_styles[sender] = self.text_processor.extract_style_features(user_messages)
        
        # 사용자 간 스타일 유사도 계산
        users = list(user_styles.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1, user2 = users[i], users[j]
                similarity = self.text_processor.calculate_style_similarity(
                    user_styles[user1],
                    user_styles[user2]
                )
                style_similarities[f"{user1}->{user2}"] = similarity
        
        result = {
            'user_styles': user_styles,
            'style_similarities': style_similarities
        }
        
        if cache_key:
            self.cache_manager.set('pos', cache_key, result)
        return result