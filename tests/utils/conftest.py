import pytest
import pandas as pd
from datetime import datetime
from src.utils.text_processor import TextProcessor
from src.utils.cache_manager import CacheManager

@pytest.fixture
def text_processor():
    return TextProcessor()

@pytest.fixture
def cache_manager():
    return CacheManager()

@pytest.fixture
def base_sample_df():
    data = {
        'datetime': [
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 12, 1),
            datetime(2024, 1, 1, 12, 2)
        ],
        'sender': ['User1', 'User2', 'User1'],
        'message': ['안녕하세요', '반갑습니다', '좋은 하루 되세요'],
        'preprocessed_message': ['안녕하세요', '반갑습니다', '좋은 하루 되세요']
    }
    return pd.DataFrame(data) 