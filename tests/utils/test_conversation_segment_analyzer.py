import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.utils.conversation_thread_analyzer import ConversationThreadAnalyzer
import asyncio

@pytest.fixture
def sample_df():
    data = {
        'datetime': [
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 12, 1),
            datetime(2024, 1, 1, 12, 2),
            datetime(2024, 1, 1, 12, 3)
        ],
        'sender': ['User1', 'User2', 'User1', 'User2'],
        'message': ['Hello', 'Hi', 'How are you?', 'Good!'],
        'preprocessed_message': ['Hello', 'Hi', 'How are you?', 'Good!']
    }
    return pd.DataFrame(data)

@pytest.fixture
def analyzer():
    return ConversationThreadAnalyzer()

@pytest.mark.asyncio
async def test_analyze_thread(analyzer, sample_df):
    chat_id = 'test_chat'
    start_time = sample_df['datetime'].min()
    end_time = sample_df['datetime'].max()
    
    result = await analyzer.analyze_thread(sample_df, start_time, end_time, chat_id)
    
    assert 'basic_stats' in result
    assert 'turn_taking' in result
    assert 'response_patterns' in result
    assert 'conversation_dynamics' in result
    assert result['chat_id'] == chat_id

def test_basic_stats(analyzer, sample_df):
    stats = analyzer._get_basic_stats(sample_df)
    assert stats['message_count'] == 4
    assert stats['participant_count'] == 2
    assert 'User1' in stats['participants']
    assert 'User2' in stats['participants']

@pytest.mark.asyncio
async def test_analyze_threads(analyzer, sample_df):
    chat_id = 'test_chat'
    thread_results = {
        'threads': [
            [sample_df['datetime'].min(), sample_df['datetime'].max()]
        ],
        'thread_stats': {
            'total_threads': 1,
            'avg_thread_length': 180,
            'max_thread_length': 180
        }
    }
    
    result = await analyzer.analyze_threads(sample_df, chat_id, thread_results)
    
    assert 'segments' in result
    assert 'thread_stats' in result
    assert len(result['segments']) > 0

@pytest.mark.asyncio
async def test_empty_thread(analyzer):
    """빈 스레드 분석 테스트"""
    empty_df = pd.DataFrame(columns=['datetime', 'sender', 'message', 'preprocessed_message'])
    chat_id = 'test_chat'
    start_time = datetime.now()
    end_time = datetime.now() + timedelta(minutes=1)
    
    result = await analyzer.analyze_thread(empty_df, start_time, end_time, chat_id)
    assert result == analyzer._get_empty_analysis_result()

@pytest.mark.asyncio
async def test_thread_metrics(analyzer, sample_df):
    """스레드 메트릭 분석 테스트"""
    result = await analyzer._analyze_thread_metrics(sample_df)
    
    assert 'turn_taking' in result
    assert 'response_patterns' in result
    assert isinstance(result['turn_taking']['avg_response_time'], float)
    assert isinstance(result['turn_taking']['turn_distribution'], dict)
    assert all(isinstance(v, float) for v in result['response_patterns'].values())

@pytest.mark.asyncio
async def test_analyze_threads_with_timeout(analyzer, sample_df):
    """스레드 분석 타임아웃 테스트"""
    chat_id = 'test_chat'
    thread_results = {
        'threads': [
            [sample_df['datetime'].min(), sample_df['datetime'].max()]
        ],
        'thread_stats': {
            'total_threads': 1,
            'avg_thread_length': 180,
            'max_thread_length': 180
        }
    }
    
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(0.001):  # 매우 짧은 타임아웃 설정
            await analyzer.analyze_threads(sample_df, chat_id, thread_results) 