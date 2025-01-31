from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime

class ThreadMetricAnalyzer(ABC):
    """대화 분석을 위한 기본 클래스"""
    
    def initialize(self) -> None:
        """분석 시작 전 메트릭 초기화"""
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """구체적인 메트릭 초기화 구현"""
        pass
    
    @abstractmethod
    def on_first_message(self, sender: str, message: str, datetime: datetime) -> None:
        """첫 메시지 처리"""
        pass
    
    @abstractmethod
    def on_speaker_change(self, prev_sender: str, curr_sender: str, 
                         message: str, time_diff: float) -> None:
        """화자 전환 처리"""
        pass
    
    @abstractmethod
    def on_continuous_message(self, sender: str, message: str) -> None:
        """연속 발화 처리"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """분석된 메트릭 반환"""
        pass 