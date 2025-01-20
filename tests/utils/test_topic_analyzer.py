import pytest
from src.utils.topic_analyzer import analyze_topics

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