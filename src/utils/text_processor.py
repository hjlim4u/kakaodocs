from transformers import AutoTokenizer, AutoModel
from konlpy.tag import Okt
from pykospacing import Spacing
import torch
import numpy as np
import re
from typing import List, Dict, Union
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from .cache_manager import CacheManager
import emoji
from urlextract import URLExtract
from .statistical_utils import adjust_outliers

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
        self.url_extractor = URLExtract()
        
        self.chat_patterns = {
            'single_consonants': r'[ㄱ-ㅎ]+',  # 자음만 있는 경우
            'single_vowels': r'[ㅏ-ㅣ]+',      # 모음만 있는 경우
            'media': r'^동영상$|^사진$|^사진 [0-9]{1,2}장$|^<(사진|동영상) 읽지 않음>$',
            # 필수적이지 않은 특수문자
            'special_chars': r'[~@#$%^&*()_+=`\[\]{}|\\<>]',
            # 시스템 메시지 패턴
            'system_messages': {
                'location': r'지도: .+',  # 위치 공유
                'map_share': r'\[네이버 지도\]',  # 지도 공유
                'audio_file': r'[a-f0-9]{64}\.m4a',  # 음성 메시지
                'music_share': r"'.+' 음악을 공유했습니다\.",  # 음악 공유
                'file_share': r'파일: .+\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|txt)$',  # 파일 공유
            }
        }
        
        self.pos_groups = {
            'substantives': ['Noun'],
            'predicates': ['Verb', 'Adjective'],
            'modifiers': ['Adverb'],
            'endings': ['Josa', 'Eomi'],
            'expressions': ['KoreanParticle', 'Exclamation']
        }

    def preprocess_message(self, message: str) -> str:
        """메시지 전처리 - 시스템 메시지, URL, 이모지, 미디어 첨부 메시지, 단일 자음/모음 제거"""
        processed_msg = message
        
        # 시스템 메시지 필터링
        for pattern in self.chat_patterns['system_messages'].values():
            if re.match(pattern, processed_msg):
                return ''
        
        # 미디어 첨부 메시지인 경우 빈 문자열 반환
        if re.match(self.chat_patterns['media'], processed_msg):
            return ''
        
        # URL 제거
        urls = self.url_extractor.find_urls(processed_msg)
        for url in urls:
            processed_msg = processed_msg.replace(url, '')
        
        # 이모지 제거
        processed_msg = emoji.replace_emoji(processed_msg, '')
        
        # 단일 자음/모음 제거
        processed_msg = re.sub(self.chat_patterns['single_consonants'], '', processed_msg)
        processed_msg = re.sub(self.chat_patterns['single_vowels'], '', processed_msg)
        
        # 불필요한 특수문자 제거
        processed_msg = re.sub(self.chat_patterns['special_chars'], '', processed_msg)
        
        # 연속된 공백 제거 및 양쪽 공백 제거
        processed_msg = ' '.join(processed_msg.split())
        
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
            
        # 전처리된 메시지에서 태그 제거 후 형태소 분석 수행
        # cleaned_messages = [self._remove_preprocessing_tags(msg) for msg in messages]
        corrected_messages = [self.spacing(msg) for msg in messages if msg]  # 빈 문자열 제외
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

    def calculate_vector_similarity(self, style1: Union[Dict, np.ndarray], style2: Union[Dict, np.ndarray]) -> float:
        """두 스타일 벡터 간의 유사도 계산"""
        # 입력이 딕셔너리인 경우 numpy 배열로 변환
        vec1 = np.array(list(style1.values())) if isinstance(style1, dict) else style1
        vec2 = np.array(list(style2.values())) if isinstance(style2, dict) else style2
        
        # 벡터가 비어있거나 크기가 다른 경우 처리
        if vec1.size == 0 or vec2.size == 0 or vec1.size != vec2.size:
            return 0.0
        
        # 코사인 유사도 계산
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

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
        
        # 이상치를 조정한 메시지 길이 평균 계산
        message_lengths = [len(msg) for msg in messages]
        adjusted_lengths = adjust_outliers(message_lengths, iqr_multiplier=3.0)
        
        # 전체 메시지에 대한 통계 추가
        total_messages = len(messages)
        total_chars = sum(message_lengths)
        
        features = {
            'lexical_features': {
                'avg_message_length': np.mean(adjusted_lengths) if total_messages > 0 else 0
                # 'total_messages': total_messages,
                # 'total_chars': total_chars,
                # 'chars_per_message': total_chars / total_messages if total_messages > 0 else 0
            },
            'morphological_features': {
                'pos_ratios': pos_ratios,
                'pos_totals': pos_totals,
                'avg_morphemes_per_message': total_morphemes / total_messages if total_messages > 0 else 0,
                # 'morpheme_complexity': np.std(sentence_lengths),
                'normalized_word_ratio': len([word for pos_list in pos_results 
                                           for word, pos in pos_list 
                                           if pos in ['Noun', 'Verb', 'Adjective']]) / total_morphemes
            },
            'syntactic_features': {
                'question_rate': sum('?' in msg for msg in messages) / total_messages if total_messages > 0 else 0,
                'exclamation_rate': sum('!' in msg for msg in messages) / total_messages if total_messages > 0 else 0,
                # 'ending_variation': len(set(pos[-1][1] for pos in pos_results if pos)) / total_messages if total_messages > 0 else 0,
                'formal_ending_ratio': len([pos for pos_list in pos_results 
                                          for word, pos in pos_list 
                                          if pos == 'Eomi' and word.endswith(('습니다', '니다'))]) / total_messages if total_messages > 0 else 0
            },
            'metadata': {
                'total_messages': total_messages,
                'total_chars': total_chars,
                'total_morphemes': total_morphemes,
                'analysis_period': {
                    'start': start_time,
                    'end': end_time
                } if start_time and end_time else None
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
            
            similarity = self.calculate_vector_similarity(vec1, vec2)
            similarities.append(similarity)
        
        return np.average(similarities, weights=weights)