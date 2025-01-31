import re
from datetime import datetime, timedelta
import pandas as pd
from src.models.chat import ChatMessage

def determine_date_pattern(first_few_lines: list[str]) -> str:
    """Determine the date pattern used in the chat file"""
    date_patterns = [
        r'(\d{4})\. (\d{1,2})\. (\d{1,2})\. (오전|오후) (\d{1,2}):(\d{2})',
        r'(\d{4})년 (\d{1,2})월 (\d{1,2})일 (오전|오후) (\d{1,2}):(\d{2})'
    ]
    
    for line in first_few_lines:
        for pattern in date_patterns:
            if re.search(pattern, line):
                return pattern
    return date_patterns[0]  # 기본값으로 첫 번째 패턴 반환

def parse_datetime(date_str: str, pattern: str) -> datetime | None:
    """Parse KakaoTalk datetime string to datetime object"""
    match = re.match(pattern, date_str)
    if match:
        year, month, day, ampm, hour, minute = match.groups()
        hour = int(hour)
        if ampm == '오후' and hour != 12:
            hour += 12
        elif ampm == '오전' and hour == 12:
            hour = 0
        return datetime(int(year), int(month), int(day), hour, int(minute))
    return None

def parse_katalk_message(line: str, date_pattern: str) -> dict | None:
    """Parse a single line of KakaoTalk message"""
    message_patterns = [
        f'({date_pattern.replace("(", "(?:")}), (.+?) : (.+)'
    ]
    
    for pattern in message_patterns:
        match = re.match(pattern, line)
        if match:
            datetime_str, sender, message = match.groups()
            parsed_datetime = parse_datetime(datetime_str, date_pattern)
            if parsed_datetime:
                return {
                    'datetime': parsed_datetime,
                    'sender': sender.strip(),
                    'message': message.strip()
                }
    return None

def distribute_seconds(messages: list[dict]) -> list[dict]:
    """
    동일 시간대 메시지들의 초 단위를 균등하게 분포
    
    Args:
        messages: 파싱된 메시지 리스트
        
    Returns:
        초 단위가 할당된 메시지 리스트
    """
    if not messages:
        return messages
        
    current_group = []
    current_minute = messages[0]['datetime'].strftime('%Y%m%d%H%M')
    
    for msg in messages:
        msg_minute = msg['datetime'].strftime('%Y%m%d%H%M')
        
        # 새로운 분으로 넘어갔을 때 이전 그룹 처리
        if msg_minute != current_minute:
            # 현재 그룹의 메시지들에 초 분배
            step = 60 / (len(current_group) + 1)  # +1로 0초와 60초 제외
            for i, group_msg in enumerate(current_group):
                new_second = int(step * (i + 1))  # 1부터 시작하여 균등 분배
                group_msg['datetime'] = group_msg['datetime'].replace(second=new_second)
            
            current_group = []
            current_minute = msg_minute
            
        current_group.append(msg)
    
    # 마지막 그룹 처리
    if current_group:
        step = 60 / (len(current_group) + 1)
        for i, group_msg in enumerate(current_group):
            new_second = int(step * (i + 1))
            group_msg['datetime'] = group_msg['datetime'].replace(second=new_second)
    
    return messages

async def parse_katalk_file(file_path: str) -> pd.DataFrame:
    """Parse KakaoTalk chat export file to DataFrame"""
    messages = []
    current_message = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 첫 10줄 정도만 읽어서 날짜 패턴 파악
        first_lines = []
        for _ in range(10):
            line = f.readline().strip()
            if line:
                first_lines.append(line)
        
        date_pattern = determine_date_pattern(first_lines)
        
        # 파일 처음으로 되돌아가기
        f.seek(0)
        
        for line in f:
            line = line.strip()
            if not line or re.match(r'\d{4}년 \d{1,2}월 \d{1,2}일 \w요일', line):
                continue
                
            parsed = parse_katalk_message(line, date_pattern)
            if parsed:
                if current_message:
                    messages.append(ChatMessage(**current_message))
                current_message = parsed
            elif current_message:
                current_message['message'] += f"\n{line}"
                
    if current_message:
        messages.append(ChatMessage(**current_message))
    
    # 초 단위 분배 적용
    messages = distribute_seconds([msg.model_dump() for msg in messages])
    
    return pd.DataFrame(messages)