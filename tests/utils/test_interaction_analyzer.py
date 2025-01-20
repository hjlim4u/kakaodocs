import pytest
import pandas as pd
from datetime import datetime
import networkx as nx
from src.utils.interaction_analyzer import InteractionAnalyzer
from src.utils.text_processor import TextProcessor

@pytest.fixture
def sample_df():
    data = {
        'datetime': [
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 12, 1),
            datetime(2024, 1, 1, 12, 2)
        ],
        'sender': ['User1', 'User2', 'User1'],
        'message': ['안녕하세요', '반갑습니다', '좋은 하루 되세요']
    }
    return pd.DataFrame(data)

@pytest.fixture
def analyzer():
    text_processor = TextProcessor()
    return InteractionAnalyzer(text_processor)

def test_preprocess_message(analyzer):
    message = "안녕하세요 ㅋㅋㅋ 😊 http://example.com"
    result = analyzer._preprocess_message(message)
    assert '[LAUGH]' in result
    assert '[EMOJI]' in result
    assert 'http://example.com' not in result

@pytest.mark.asyncio
async def test_analyze_interactions(analyzer, sample_df):
    # 테스트용 thread_results 생성
    thread_results = {
        'threads': [
            [sample_df['datetime'].min(), sample_df['datetime'].max()]
        ],
        'thread_stats': {
            'total_threads': 1,
            'avg_thread_length': 120,
            'max_thread_length': 120
        }
    }
    
    # chat_id 추가
    chat_id = 'test_chat'
    
    result = await analyzer.analyze_interactions(sample_df, chat_id, thread_results)
    
    # 기본 검증
    assert isinstance(result['network'], nx.Graph)
    assert 'centrality' in result
    assert 'density' in result
    assert 'thread_stats' in result
    assert 'language_style_analysis' in result
    
    # 언어 스타일 분석 결과 검증
    style_analysis = result['language_style_analysis']
    assert 'user_styles' in style_analysis
    assert 'style_similarities' in style_analysis

def test_extract_style_features(analyzer):
    messages = pd.Series(['안녕하세요!', '잘 지내시나요? ㅎㅎ'])
    result = analyzer._extract_style_features(messages)
    assert 'lexical_features' in result
    assert 'morphological_features' in result
    assert 'syntactic_features' in result

def test_analyze_language_style(analyzer, sample_df):
    chat_id = 'test_chat'
    result = analyzer._analyze_language_style(sample_df, chat_id)
    
    assert isinstance(result, dict)
    assert 'user_styles' in result
    assert 'style_similarities' in result 