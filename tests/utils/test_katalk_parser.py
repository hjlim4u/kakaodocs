import pytest
from datetime import datetime
import pandas as pd
from src.utils.katalk_parser import (
    determine_date_pattern,
    parse_datetime,
    parse_katalk_message,
    distribute_seconds
)

@pytest.fixture
def sample_messages():
    return [
        {
            'datetime': datetime(2024, 1, 1, 12, 0),
            'sender': 'User1',
            'message': 'Hello'
        },
        {
            'datetime': datetime(2024, 1, 1, 12, 0),
            'sender': 'User2',
            'message': 'Hi'
        },
        {
            'datetime': datetime(2024, 1, 1, 12, 0),
            'sender': 'User3',
            'message': 'Hey'
        }
    ]

def test_determine_date_pattern():
    lines = [
        "2024. 1. 1. 오후 2:30, User1 : Hello",
        "2024년 1월 1일 오전 10:15, User2 : Hi"
    ]
    pattern = determine_date_pattern(lines)
    assert pattern in [
        r'(\d{4})\. (\d{1,2})\. (\d{1,2})\. (오전|오후) (\d{1,2}):(\d{2})',
        r'(\d{4})년 (\d{1,2})월 (\d{1,2})일 (오전|오후) (\d{1,2}):(\d{2})'
    ]

def test_parse_datetime():
    date_str = "2024. 1. 1. 오후 2:30"
    pattern = r'(\d{4})\. (\d{1,2})\. (\d{1,2})\. (오전|오후) (\d{1,2}):(\d{2})'
    result = parse_datetime(date_str, pattern)
    assert isinstance(result, datetime)
    assert result.hour == 14
    assert result.minute == 30

def test_parse_katalk_message():
    line = "2024. 1. 1. 오후 2:30, User1 : Hello"
    pattern = r'(\d{4})\. (\d{1,2})\. (\d{1,2})\. (오전|오후) (\d{1,2}):(\d{2})'
    result = parse_katalk_message(line, pattern)
    assert result['sender'] == 'User1'
    assert result['message'] == 'Hello'

def test_distribute_seconds(sample_messages):
    result = distribute_seconds(sample_messages)
    assert len(result) == 3
    # Check if seconds are distributed around 30
    seconds = [msg['datetime'].second for msg in result]
    assert all(20 <= s <= 40 for s in seconds)
    # Check if all seconds are different
    assert len(set(seconds)) == 3 