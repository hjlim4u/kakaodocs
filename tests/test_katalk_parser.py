import pytest
from datetime import datetime
from src.utils.katalk_parser import parse_katalk_message, determine_date_pattern, distribute_seconds

# 테스트에서 사용할 기본 날짜 패턴
DATE_PATTERN = r'(\d{4})\. (\d{1,2})\. (\d{1,2})\. (오전|오후) (\d{1,2}):(\d{2})'

def test_parse_basic_message():
    line = "2021. 4. 23. 오후 3:01, 임형주 : 도착함?"
    result = parse_katalk_message(line, DATE_PATTERN)
    
    assert result is not None
    assert result["datetime"] == datetime(2021, 4, 23, 15, 1)
    assert result["sender"] == "임형주"
    assert result["message"] == "도착함?"

def test_parse_message_with_media():
    media_messages = [
        "2021. 4. 23. 오후 3:26, 임형주 : 사진 12장",
        "2021. 8. 12. 오후 6:55, 임형주 : 동영상",
        "2021. 8. 22. 오후 9:42, 치훈 : 사진 4장"
    ]
    
    for msg in media_messages:
        result = parse_katalk_message(msg, DATE_PATTERN)
        assert result is not None
        assert "사진" in result["message"] or "동영상" in result["message"]

def test_parse_message_with_url():
    # Test message containing URL
    line = "2021. 9. 15. 오전 10:01, 치훈 : https://www.apple.com/kr/ipad-mini/"
    result = parse_katalk_message(line, DATE_PATTERN)
    
    assert result is not None
    assert result["datetime"] == datetime(2021, 9, 15, 10, 1)
    assert result["sender"] == "치훈"
    assert "https://" in result["message"]

def test_parse_invalid_messages():
    # Test invalid message formats
    invalid_messages = [
        "",  # Empty string
        "2021년 4월 23일 금요일",  # Date header
        "Invalid message format",  # Random text
        "2021. 4. 23."  # Incomplete timestamp
    ]
    
    for msg in invalid_messages:
        result = parse_katalk_message(msg, DATE_PATTERN)
        assert result is None

def test_parse_message_with_special_characters():
    # Test messages containing special characters
    line = "2021. 10. 1. 오후 7:47, 임형주 : ㅋㅋㅋㅋ 근데 시니어는 저렇게 코딩은 안한다. 시니어도 stack overflow에 있는거 보면서 함"
    result = parse_katalk_message(line, DATE_PATTERN)
    
    assert result is not None
    assert result["datetime"] == datetime(2021, 10, 1, 19, 47)
    assert result["sender"] == "임형주"
    assert "ㅋㅋㅋㅋ" in result["message"]

def test_parse_message_time_conversion():
    # Test AM/PM time conversion
    am_message = "2021. 9. 15. 오전 10:01, 치훈 : 테스트"
    pm_message = "2021. 9. 15. 오후 10:01, 치훈 : 테스트"
    
    am_result = parse_katalk_message(am_message, DATE_PATTERN)
    pm_result = parse_katalk_message(pm_message, DATE_PATTERN)
    
    assert am_result["datetime"].hour == 10
    assert pm_result["datetime"].hour == 22

def test_parse_deleted_message():
    # Test deleted message format
    line = "2021. 10. 25. 오후 10:11, 치훈 : 삭제된 메시지입니다."
    result = parse_katalk_message(line, DATE_PATTERN)
    
    assert result is not None
    assert result["datetime"] == datetime(2021, 10, 25, 22, 11)
    assert result["sender"] == "치훈"
    assert result["message"] == "삭제된 메시지입니다." 

def test_distribute_seconds():
    # 동일 시간대 메시지들 테스트
    messages = [
        {
            "datetime": datetime(2021, 4, 23, 15, 1),
            "sender": "임형주",
            "message": "메시지1"
        },
        {
            "datetime": datetime(2021, 4, 23, 15, 1),
            "sender": "치훈",
            "message": "메시지2"
        },
        {
            "datetime": datetime(2021, 4, 23, 15, 1),
            "sender": "임형주",
            "message": "메시지3"
        }
    ]
    
    distributed = distribute_seconds(messages)
    
    # 검증
    assert len(distributed) == 3
    seconds = [msg["datetime"].second for msg in distributed]
    
    # 초 단위가 30초를 중심으로 분포되어 있는지 확인
    assert 20 <= seconds[0] <= 40  # 첫 번째 메시지
    assert 20 <= seconds[1] <= 40  # 두 번째 메시지
    assert 20 <= seconds[2] <= 40  # 세 번째 메시지
    
    # 메시지 간 간격이 동일한지 확인
    diff1 = seconds[1] - seconds[0]
    diff2 = seconds[2] - seconds[1]
    assert abs(diff1 - diff2) <= 1  # 부동소수점 오차 고려

def test_distribute_seconds_single_message():
    # 단일 메시지 테스트
    messages = [{
        "datetime": datetime(2021, 4, 23, 15, 1),
        "sender": "임형주",
        "message": "메시지1"
    }]
    
    distributed = distribute_seconds(messages)
    
    # 단일 메시지는 정확히 30초에 위치
    assert distributed[0]["datetime"].second == 30 