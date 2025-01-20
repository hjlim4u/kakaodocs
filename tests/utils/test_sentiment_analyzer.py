import pytest
from src.utils.sentiment_analyzer import AdvancedSentimentAnalyzer

@pytest.fixture
def analyzer():
    return AdvancedSentimentAnalyzer()

@pytest.mark.asyncio
async def test_analyze_emotion(analyzer):
    text = "오늘 날씨가 정말 좋네요!"
    result = await analyzer.analyze_emotion(text)
    
    # 기본 감정 상태 확인
    assert isinstance(result, dict)
    assert 'positive' in result
    assert 'negative' in result
    assert 'neutral' in result
    
    # 값 범위 확인
    assert all(isinstance(v, float) for v in result.values())
    assert all(0 <= v <= 1 for v in result.values())
    
    # 확률 합계가 1에 근접
    total = sum(result.values())
    assert 0.99 <= total <= 1.01 

@pytest.mark.asyncio
async def test_analyze_sentiments(analyzer, sample_df):
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
    
    result = await analyzer.analyze_sentiments(sample_df, chat_id, thread_results)
    
    assert 'user_sentiments' in result
    assert 'sentiment_flow' in result
    assert 'overall_sentiment' in result
    assert result['chat_id'] == chat_id 