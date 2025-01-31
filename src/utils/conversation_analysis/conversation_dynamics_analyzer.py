from collections import defaultdict
from typing import Dict
import networkx as nx
import numpy as np
from .base_analyzer import ThreadMetricAnalyzer
from datetime import datetime

class ConversationDynamicsAnalyzer(ThreadMetricAnalyzer):
    def _initialize(self) -> None:
        self.G = nx.DiGraph()
        self.participant_messages = defaultdict(int)
    
    def on_first_message(self, sender: str, message: str, datetime: datetime) -> None:
        self.participant_messages[sender] += 1
    
    def on_speaker_change(self, prev_sender: str, curr_sender: str, 
                         message: str, time_diff: float) -> None:
        self.participant_messages[curr_sender] += 1
        
        # 대화 흐름 그래프 업데이트
        if self.G.has_edge(prev_sender, curr_sender):
            self.G[prev_sender][curr_sender]['weight'] += 1
        else:
            self.G.add_edge(prev_sender, curr_sender, weight=1)
    
    def on_continuous_message(self, sender: str, message: str) -> None:
        self.participant_messages[sender] += 1
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        """
        지니 계수 계산 (0: 완벽한 평등, 1: 완벽한 불평등)
        """
        if len(values) < 2:  # 참여자가 1명 이하면 불평등을 측정할 수 없음
            return 0.0
        
        # 빈 값이나 모든 값이 0인 경우 처리
        total = np.sum(values)
        if total == 0:
            return 0.0
        
        # 오름차순 정렬
        values = np.sort(values)
        n = len(values)
        
        # 누적 합 계산
        cumsum = np.cumsum(values)
        
        # 지니 계수 계산
        # sum(2i - n - 1) * xi / (n * total)
        indices = np.arange(1, n + 1)
        gini = ((2 * indices - n - 1) * values).sum() / (n * total)
        
        return float(max(0.0, min(1.0, gini)))  # 0과 1 사이의 값으로 제한
    
    def get_metrics(self) -> Dict:
        centrality_measures = {
            'in_degree': {},
            'out_degree': {},
        }
        
        if len(self.G) > 0:
            centrality_measures = {
                'in_degree': nx.in_degree_centrality(self.G),  # 대화를 받는 정도
                'out_degree': nx.out_degree_centrality(self.G),  # 대화를 시작하는 정도
                'pagerank': nx.pagerank(self.G),  # 전반적인 대화 영향력
                # 추가 중심성 지표들 (필요시 주석 해제)
                # 'betweenness': nx.betweenness_centrality(self.G),
                # 'eigenvector': nx.eigenvector_centrality_numpy(self.G),
                'closeness': nx.closeness_centrality(self.G),
            }
            
            # 전체 네트워크 수준의 지표들
            network_metrics = {
                'density': nx.density(self.G),  # 네트워크 밀도
                'reciprocity': nx.reciprocity(self.G),  # 양방향 대화 비율
                'clustering_coefficient': nx.average_clustering(self.G),
            }
            
            # 대화 패턴 분석
            edge_weights = [d['weight'] for (u, v, d) in self.G.edges(data=True)]
            interaction_metrics = {
                'total_interactions': sum(edge_weights),  # 전체 대화 상호작용 수
                'avg_interaction_strength': np.mean(edge_weights),  # 평균 상호작용 강도
                # 'interaction_variance': np.var(edge_weights),
            }
            
            return {
                'participation_inequality': self._calculate_gini(
                    np.array(list(self.participant_messages.values()))
                ),
                'flow_centrality': centrality_measures,
                'network_metrics': network_metrics,
                'interaction_metrics': interaction_metrics,
            }

    def merge_metrics(self, metrics: Dict) -> Dict:
        pass
