from typing import Any, Dict, Optional
import sys
from datetime import datetime, timedelta
import hashlib

class CacheManager:
    def __init__(self, max_size_bytes: int = 100 * 1024 * 1024):
        self._caches: Dict[str, Dict] = {
            'thread': {},      # 스레드 캐시
            'embedding': {},   # 임베딩 캐시
            'pos': {},        # 형태소 분석 캐시
            'participants': {} # 참여자 목록 캐시
        }
        self._max_size_bytes = max_size_bytes
        self._last_access_times: Dict[str, Dict[str, datetime]] = {
            cache_type: {} for cache_type in self._caches.keys()
        }
        
    def generate_cache_key(self, cache_type: str, **kwargs) -> str:
        """캐시 키 생성
        Args:
            cache_type: 캐시 타입 ('thread', 'embedding', 'pos', 'participants')
            **kwargs: 키 생성에 사용될 파라미터들
                - chat_id: 채팅방 ID
                - sender: 발신자
                - start_time: 시작 시간 (str)
                - end_time: 종료 시간 (str)
                - message: 메시지 내용
                - suffix: 추가 식별자
        """
        if cache_type not in self._caches:
            raise ValueError(f"Invalid cache type: {cache_type}")

        key_parts = []
        
        # 채팅방 ID가 있는 경우
        if 'chat_id' in kwargs:
            key_parts.append(str(kwargs['chat_id']))
        
        # 발신자가 있는 경우
        if 'sender' in kwargs:
            key_parts.append(str(kwargs['sender']))
            
        # 시간 범위가 있는 경우
        if 'start_time' in kwargs and 'end_time' in kwargs:
            key_parts.append(f"{kwargs['start_time']}_{kwargs['end_time']}")
            
        # 메시지 내용이 있는 경우 (임베딩 캐시용)
        if 'message' in kwargs:
            # 긴 메시지는 해시로 변환
            msg_hash = hashlib.md5(kwargs['message'].encode()).hexdigest()
            key_parts.append(msg_hash)
            
        # 추가 식별자가 있는 경우
        if 'suffix' in kwargs:
            key_parts.append(str(kwargs['suffix']))
            
        # 키 조합
        if not key_parts:
            raise ValueError("No valid parameters provided for key generation")
            
        return f"{cache_type}:{':'.join(key_parts)}"

    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if cache_type not in self._caches:
            return None
            
        if key in self._caches[cache_type]:
            self._last_access_times[cache_type][key] = datetime.now()
            return self._caches[cache_type][key]
        return None
        
    def set(self, cache_type: str, key: str, value: Any) -> None:
        """캐시에 값 저장"""
        if cache_type not in self._caches:
            return
            
        self._clear_if_needed()
        self._caches[cache_type][key] = value
        self._last_access_times[cache_type][key] = datetime.now()
        
    def _clear_if_needed(self) -> None:
        """메모리 사용량 체크 및 캐시 정리"""
        total_size = sum(sys.getsizeof(cache) for cache in self._caches.values())
        
        if total_size > self._max_size_bytes:
            self._evict_entries()
            
    def _evict_entries(self) -> None:
        """LRU 정책으로 캐시 엔트리 제거"""
        all_entries = []
        for cache_type in self._caches:
            for key in self._caches[cache_type]:
                all_entries.append((
                    cache_type,
                    key,
                    self._last_access_times[cache_type][key]
                ))
        
        # 가장 오래된 항목부터 정렬
        all_entries.sort(key=lambda x: x[2])
        
        # 50%의 오래된 항목 제거
        entries_to_remove = len(all_entries) // 2
        for cache_type, key, _ in all_entries[:entries_to_remove]:
            del self._caches[cache_type][key]
            del self._last_access_times[cache_type][key] 