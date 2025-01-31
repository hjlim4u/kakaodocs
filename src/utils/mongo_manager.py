from motor.motor_asyncio import AsyncIOMotorClient
from typing import Dict, List
from datetime import datetime

class MongoManager:
    def __init__(self, connection_string: str = "mongodb://localhost:27017"):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client.chat_analysis
        self.thread_collection = self.db.thread_analyses
        
    async def store_thread_analyses(self, chat_id: str, user_id: int, analyses: List[Dict]) -> None:
        """대화 구간 분석 결과 저장"""
        for analysis in analyses:
            # 기존 문서 검색을 위한 필터
            filter_query = {
                "chat_id": chat_id,
                "user_id": user_id,
                "period.start_time": analysis["period"]["start_time"],
                "period.end_time": analysis["period"]["end_time"]
            }
            
            # 저장할 문서 준비
            document = {
                "chat_id": chat_id,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                **analysis
            }
            
            # upsert 수행 (없으면 삽입, 있으면 업데이트)
            await self.thread_collection.update_one(
                filter_query,
                {"$set": document},
                upsert=True
            )
    
    async def get_thread_analyses(self, chat_id: str, user_id: int) -> List[Dict]:
        """특정 채팅방의 대화 구간 분석 결과 조회"""
        cursor = self.thread_collection.find({
            "chat_id": chat_id,
            "user_id": user_id
        }).sort("period.start_time", 1)
        
        return await cursor.to_list(length=None) 