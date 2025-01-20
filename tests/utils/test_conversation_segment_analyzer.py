import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.utils.conversation_thread_analyzer import ConversationThreadAnalyzer

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
        'message': ['Hello', 'Hi', 'How are you?', 'Good!']
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