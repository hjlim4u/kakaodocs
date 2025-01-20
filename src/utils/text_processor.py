from transformers import AutoTokenizer, AutoModel
from konlpy.tag import Okt
from pykospacing import Spacing
import torch
import numpy as np
import re
from typing import List, Dict
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from utils.cache_manager import CacheManager

class TextProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.model = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022",
                                             output_hidden_states=True,
                                             return_dict=True)
        self.model.eval()
        self.pos_tagger = Okt()
        self.spacing = Spacing()
        self.cache_manager = CacheManager()
        
        self.chat_patterns = {
            'emoticons': r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]',
            'chat_specific': r'(ㅋㅋ+|ㅎㅎ+|ㅠㅠ+|ㄷㄷ+)',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        }
        
        self.pos_groups = {
            'substantives': ['Noun'],
            'predicates': ['Verb', 'Adjective'],
            'modifiers': ['Adverb'],
            'endings': ['Josa', 'Eomi'],
            'expressions': ['KoreanParticle', 'Exclamation']
        }

    def preprocess_message(self, message: str) -> str:
        processed_msg = message
        processed_msg = re.sub(self.chat_patterns['urls'], '[URL]', processed_msg)
        processed_msg = re.sub(self.chat_patterns['emoticons'], '[EMOJI]', processed_msg)
        processed_msg = re.sub(r'ㅋㅋ+', '[LAUGH]', processed_msg)
        processed_msg = re.sub(r'ㅎㅎ+', '[SMILE]', processed_msg)
        processed_msg = re.sub(r'ㅠㅠ+', '[CRY]', processed_msg)
        return processed_msg.strip()

    async def get_embeddings(self, messages: List[str]) -> np.ndarray:
        embeddings = []
        for msg in messages:
            # 캐시에서 임베딩 확인
            try:
                cache_key = self.cache_manager.generate_cache_key(
                    'embedding',
                    message=msg
                )
            except ValueError:
                # 키 생성 실패 시 캐시 사용하지 않음
                cache_key = None
            
            if cache_key:
                cached_embedding = self.cache_manager.get('embedding', cache_key)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    continue
            
            processed_msg = self.preprocess_message(msg)
            if not processed_msg:
                embedding = np.zeros(768)
                embeddings.append(embedding)
                continue
            
            inputs = self.tokenizer(processed_msg, 
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                embedding = torch.mean(last_hidden_state, dim=1).numpy()[0]
                
                if cache_key:
                    self.cache_manager.set('embedding', cache_key, embedding)
                embeddings.append(embedding)
        
        return np.array(embeddings)

    def analyze_morphology(self, messages: List[str], chat_id: str = None, sender: str = None, 
                          start_time: str = None, end_time: str = None) -> Dict:
        """형태소 분석 수행
        Args:
            messages: 분석할 메시지 리스트
            chat_id: 채팅방 ID
            sender: 발신자 이름
            start_time: 구간 시작 시점 (YYYY-MM-DD HH:MM:SS)
            end_time: 구간 종료 시점 (YYYY-MM-DD HH:MM:SS)
        """
        try:
            cache_key = self.cache_manager.generate_cache_key(
                'pos',
                chat_id=chat_id,
                sender=sender,
                start_time=start_time,
                end_time=end_time,
                suffix='_'.join(messages) if not (chat_id and sender) else None
            )
        except ValueError:
            cache_key = None
        
        if cache_key:
            cached_result = self.cache_manager.get('pos', cache_key)
            if cached_result is not None:
                return cached_result
            
        corrected_messages = [self.spacing(msg) for msg in messages]
        pos_results = [self.pos_tagger.pos(msg, norm=True, stem=True) 
                      for msg in corrected_messages]
        
        text = ' '.join(corrected_messages)
        total_morphemes = sum(len(pos) for pos in pos_results)
        
        # 품사별 개수 계산
        pos_totals = defaultdict(int)
        for pos_list in pos_results:
            for word, pos in pos_list:
                for group_name, pos_tags in self.pos_groups.items():
                    if pos in pos_tags:
                        pos_totals[group_name] += 1
        
        result = {
            'pos_results': pos_results,
            'text': text,
            'total_morphemes': total_morphemes,
            'pos_totals': dict(pos_totals)  # pos_ratios 대신 pos_totals 사용
        }
        
        if chat_id and sender:
            result.update({
                'chat_id': chat_id,
                'sender': sender,
                'period': {
                    'start': start_time,
                    'end': end_time
                } if start_time and end_time else None
            })
        
        if cache_key:
            self.cache_manager.set('pos', cache_key, result)
        
        return result

    def _calculate_pos_ratios(self, pos_results: List, total_morphemes: int) -> Dict:
        pos_ratios = defaultdict(float)
        for pos_list in pos_results:
            for word, pos in pos_list:
                for group_name, pos_tags in self.pos_groups.items():
                    if pos in pos_tags:
                        pos_ratios[group_name] += 1
        return {k: v/total_morphemes for k, v in pos_ratios.items()}

    def calculate_semantic_similarity(self, 
                                   embedding1: np.ndarray, 
                                   embedding2: np.ndarray,
                                   context_window: List[np.ndarray] = None) -> float:
        """의미적 유사성 계산 - 문맥 고려"""
        direct_similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        
        if context_window:
            context_embedding = np.mean(context_window, axis=0)
            context_similarity = cosine_similarity(
                embedding2.reshape(1, -1),
                context_embedding.reshape(1, -1)
            )[0][0]
            return 0.7 * direct_similarity + 0.3 * context_similarity
        
        return direct_similarity

    def extract_style_features(self, messages: List[str], sender: str = None, chat_id: str = None,
                             start_time: str = None, end_time: str = None) -> Dict:
        """사용자의 언어 스타일 특성 추출"""
        # 형태소 분석 수행
        morphology_result = self.analyze_morphology(
            messages, 
            chat_id=chat_id, 
            sender=sender,
            start_time=start_time,
            end_time=end_time
        )
        
        pos_results = morphology_result['pos_results']
        text = morphology_result['text']
        total_morphemes = morphology_result['total_morphemes']
        pos_totals = morphology_result['pos_totals']
        
        # pos_totals를 pos_ratios로 변환
        pos_ratios = {k: v/total_morphemes for k, v in pos_totals.items()}
        
        # 문장 구조 특성
        sentence_lengths = [len(pos) for pos in pos_results]
        
        features = {
            'lexical_features': {
                'avg_message_length': np.mean([len(msg) for msg in messages]),
                'vocab_diversity': len(set(text.split())) / len(text.split()) if text else 0,
                'emoji_usage_rate': len(re.findall(self.chat_patterns['emoticons'], text)) / len(messages),
                'chat_specific_usage': len(re.findall(self.chat_patterns['chat_specific'], text)) / len(messages)
            },
            'morphological_features': {
                'pos_ratios': pos_ratios,  # 변환된 pos_ratios 사용
                'pos_totals': pos_totals,  # 원본 pos_totals 추가
                'avg_morphemes_per_message': total_morphemes / len(messages),
                'morpheme_complexity': np.std(sentence_lengths),
                'normalized_word_ratio': len([word for pos_list in pos_results 
                                           for word, pos in pos_list 
                                           if pos in ['Noun', 'Verb', 'Adjective']]) / total_morphemes
            },
            'syntactic_features': {
                'question_rate': sum('?' in msg for msg in messages) / len(messages),
                'exclamation_rate': sum('!' in msg for msg in messages) / len(messages),
                'ending_variation': len(set(pos[-1][1] for pos in pos_results if pos)) / len(messages),
                'formal_ending_ratio': len([pos for pos_list in pos_results 
                                          for word, pos in pos_list 
                                          if pos == 'Eomi' and word.endswith(('습니다', '니다'))]) / len(messages)
            }
        }
        return features

    def calculate_style_similarity(self, style1: Dict, style2: Dict) -> float:
        """두 사용자 간의 언어 스타일 유사도 계산"""
        similarities = []
        weights = [0.3, 0.4, 0.3]  # 어휘, 형태소, 구문 특성의 가중치
        
        # 각 특성별 유사도 계산
        for feature_type in ['lexical_features', 'morphological_features', 'syntactic_features']:
            if feature_type == 'morphological_features':
                # 형태소 특성은 pos_ratios만 비교
                vec1 = np.array(list(style1[feature_type]['pos_ratios'].values()))
                vec2 = np.array(list(style2[feature_type]['pos_ratios'].values()))
            else:
                vec1 = np.array(list(style1[feature_type].values()))
                vec2 = np.array(list(style2[feature_type].values()))
            
            similarity = self._calculate_feature_similarity(vec1, vec2)
            similarities.append(similarity)
        
        return np.average(similarities, weights=weights)

    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """특성 벡터 간 유사도 계산"""
        if features1 is None or features2 is None or \
           features1.size == 0 or features2.size == 0 or \
           np.all(features1 == 0) or np.all(features2 == 0):
            return 0.0
        
        # 1차원 배열을 2차원으로 변환
        features1 = features1.reshape(1, -1)
        features2 = features2.reshape(1, -1)
        
        try:
            return float(cosine_similarity(features1, features2)[0][0])
        except Exception:
            return 0.0
