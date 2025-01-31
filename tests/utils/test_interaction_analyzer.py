import pytest
import pandas as pd
from datetime import datetime
import networkx as nx
from src.utils.interaction_analyzer import InteractionAnalyzer
from src.utils.text_processor import TextProcessor
import asyncio
import psutil
import os

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

@pytest.mark.asyncio
async def test_analyze_interactions_with_invalid_input(analyzer):
    """잘못된 입력에 대한 예외 처리 테스트"""
    with pytest.raises(ValueError):
        await analyzer.analyze_interactions(
            pd.DataFrame(),  # 빈 데이터프레임
            'test_chat'
        )

@pytest.mark.asyncio
async def test_analyze_interactions_with_single_user(analyzer, sample_df):
    """단일 사용자 대화 분석 테스트"""
    single_user_df = sample_df.copy()
    single_user_df['sender'] = 'User1'
    result = await analyzer.analyze_interactions(single_user_df, 'test_chat')
    
    assert result['language_style_analysis']['style_similarities'] == {}
    assert result['network'].number_of_nodes() == 1
    assert result['density'] == 0

@pytest.mark.asyncio
async def test_style_feature_extraction_with_empty_messages(analyzer):
    """빈 메시지에 대한 스타일 특성 추출 테스트"""
    empty_messages = pd.Series(['', ' ', '  '])
    result = analyzer._extract_style_features(empty_messages)
    
    assert result['lexical_features']['avg_length'] == 0
    assert result['morphological_features']['pos_ratios'] == {}
    assert result['syntactic_features']['question_rate'] == 0 

@pytest.mark.asyncio
async def test_concurrent_interaction_analysis(analyzer):
    """동시 분석 처리 테스트"""
    chat_ids = [f'test_chat_{i}' for i in range(3)]
    dfs = [sample_df.copy() for _ in range(3)]
    
    tasks = [
        analyzer.analyze_interactions(df, chat_id)
        for df, chat_id in zip(dfs, chat_ids)
    ]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 3
    assert all('language_style_analysis' in r for r in results)

@pytest.mark.asyncio
async def test_memory_usage(analyzer, sample_df):
    """메모리 사용량 테스트"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # 대량의 데이터로 테스트
    large_df = pd.concat([sample_df] * 1000, ignore_index=True)
    await analyzer.analyze_interactions(large_df, 'test_chat')
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
    assert memory_increase < 500  # 메모리 증가가 500MB 이하 