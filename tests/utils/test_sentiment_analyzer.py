import pytest
from src.utils.sentiment_analyzer import AdvancedSentimentAnalyzer
import asyncio
import pandas as pd
import time

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

@pytest.mark.asyncio
async def test_analyze_emotion_detailed(analyzer):
    """상세 감정 분석 테스트"""
    test_cases = [
        ("정말 행복해요!", {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}),
        ("너무 실망했어요", {'positive': 0.1, 'negative': 0.8, 'neutral': 0.1}),
        ("오늘 날씨입니다", {'positive': 0.3, 'negative': 0.3, 'neutral': 0.4})
    ]
    
    for text, expected in test_cases:
        result = await analyzer.analyze_emotion(text)
        for key in expected:
            assert abs(result[key] - expected[key]) < 0.2

@pytest.mark.asyncio
async def test_analyze_sentiments_with_cache(analyzer, sample_df):
    """감정 분석 캐시 동작 테스트"""
    chat_id = 'test_chat'
    thread_results = {
        'threads': [[sample_df['datetime'].min(), sample_df['datetime'].max()]],
        'thread_stats': {'total_threads': 1, 'avg_thread_length': 180}
    }
    
    # 첫 번째 분석
    result1 = await analyzer.analyze_sentiments(sample_df, chat_id, thread_results)
    # 캐시된 결과 사용
    result2 = await analyzer.analyze_sentiments(sample_df, chat_id, thread_results)
    
    assert result1 == result2 

@pytest.mark.asyncio
async def test_model_loading(analyzer):
    """모델 로딩 테스트"""
    assert analyzer.tokenizer is not None
    assert analyzer.model is not None
    assert analyzer.model.config.num_labels == 2

@pytest.mark.asyncio
async def test_batch_sentiment_analysis(analyzer):
    """배치 감정 분석 테스트"""
    messages = [
        "정말 행복한 하루였어요!",
        "너무 슬픈 일이에요...",
        "오늘 날씨가 좋네요"
    ]
    results = await asyncio.gather(*[analyzer.analyze_emotion(msg) for msg in messages])
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)

@pytest.mark.asyncio
async def test_sentiment_analysis_performance(analyzer, base_sample_df):
    """대량 데이터 감정 분석 성능 테스트"""
    large_df = pd.concat([base_sample_df] * 100, ignore_index=True)
    chat_id = 'test_chat'
    thread_results = {
        'threads': [[large_df['datetime'].min(), large_df['datetime'].max()]],
        'thread_stats': {'total_threads': 1, 'avg_thread_length': 180}
    }
    
    start_time = time.time()
    result = await analyzer.analyze_sentiments(large_df, chat_id, thread_results)
    end_time = time.time()
    
    assert (end_time - start_time) < 30  # 30초 이내 처리
    assert 'sentiment_flow' in result 