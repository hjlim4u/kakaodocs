from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

async def analyze_topics(messages: list[str], eps: float = 0.3, min_samples: int = 3) -> dict:
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(messages)
    
    # DBSCAN 클러스터링
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = clustering.fit_predict(tfidf_matrix)
    
    # 각 클러스터의 주요 키워드 추출
    topics = {}
    for cluster_id in set(clusters):
        if cluster_id != -1:  # 노이즈 제외
            cluster_docs = np.where(clusters == cluster_id)[0]
            centroid = tfidf_matrix[cluster_docs].mean(axis=0)
            top_terms = np.argsort(centroid.toarray().flatten())[-5:]
            topics[f"topic_{cluster_id}"] = [
                vectorizer.get_feature_names_out()[i] 
                for i in top_terms
            ]
    
    return topics 