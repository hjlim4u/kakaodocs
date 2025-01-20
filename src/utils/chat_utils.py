from typing import List
import pandas as pd

def generate_chat_id(participants: List[str]) -> str:
    """
    채팅방 참여자들을 기반으로 고유 ID 생성
    
    Args:
        participants: 채팅방 참여자 목록
        
    Returns:
        str: 정렬된 참여자 목록을 '_'로 연결한 채팅방 ID
    """
    return '_'.join(sorted(participants))

def get_chat_id_from_df(df: pd.DataFrame) -> str:
    """
    DataFrame에서 채팅방 ID 생성
    
    Args:
        df: 채팅 데이터가 포함된 DataFrame (sender 컬럼 필요)
        
    Returns:
        str: 채팅방 ID
    """
    participants = df['sender'].unique().tolist()
    return generate_chat_id(participants) 