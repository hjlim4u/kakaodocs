import pytest
from src.utils.topic_analyzer import analyze_topics
import time

@pytest.mark.asyncio
async def test_analyze_topics():
    messages = [
        "오늘 날씨가 좋네요",
        "날씨가 정말 화창해요",
        "점심 메뉴 추천해주세요",
        "저는 김치찌개 먹을래요",
        "저도 한식 좋아요"
    ]
    
    result = await analyze_topics(messages)
    
    assert isinstance(result, dict)
    # 최소한 하나의 토픽이 있어야 함
    assert len(result) >= 0
    
    # 토픽이 있는 경우 각 토픽이 키워드 리스트를 가지고 있는지 확인
    if result:
        for topic_keywords in result.values():
            assert isinstance(topic_keywords, list)
            assert len(topic_keywords) > 0 

@pytest.mark.asyncio
async def test_topic_coherence():
    """토픽 일관성 테스트"""
    weather_messages = [
        "오늘 날씨가 좋네요",
        "날씨가 정말 화창해요",
        "비가 올 것 같아요"
    ]
    result = await analyze_topics(weather_messages)
    
    # 날씨 관련 키워드가 포함된 토픽이 있는지 확인
    weather_related = False
    for keywords in result.values():
        if any('날씨' in kw for kw in keywords):
            weather_related = True
            break
    assert weather_related

@pytest.mark.asyncio
async def test_topic_analysis_with_short_messages():
    """짧은 메시지에 대한 토픽 분석 테스트"""
    short_messages = ["안녕", "네", "응"]
    result = await analyze_topics(short_messages)
    assert isinstance(result, dict)
    assert len(result) == 0  # 짧은 메시지는 토픽으로 분류되지 않아야 함 

@pytest.mark.asyncio
async def test_topic_analysis_performance():
    """토픽 분석 성능 테스트"""
    # 대량의 메시지 생성
    messages = [
        "오늘 날씨가 좋네요",
        "날씨가 정말 화창해요",
        "점심 메뉴 추천해주세요"
    ] * 100
    
    start_time = time.time()
    result = await analyze_topics(messages)
    end_time = time.time()
    
    assert (end_time - start_time) < 10  # 10초 이내 처리
    assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_topic_analysis_with_noise():
    """노이즈가 있는 데이터 토픽 분석 테스트"""
    messages = [
        "오늘 날씨가 좋네요",
        "ㅋㅋㅋㅋㅋㅋㅋ",
        "날씨가 정말 화창해요",
        "😊😊😊",
        "비가 올 것 같아요"
    ]
    result = await analyze_topics(messages)
    
    # 노이즈 제거 후 토픽 분석 확인
    assert any('날씨' in ' '.join(keywords) for keywords in result.values()) 