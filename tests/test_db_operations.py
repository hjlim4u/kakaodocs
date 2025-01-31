import pytest
from datetime import datetime, UTC
import os
from src.utils.db_manager import DatabaseManager
from src.database.models import User, ChatMessage
from sqlalchemy import select, delete
from src.config import Settings, get_settings
import pytest_asyncio

# 테스트용 설정 로드
@pytest.fixture
def test_settings():
    # 테스트용 환경 설정
    os.environ['ENV_FILE'] = '.env.test'
    from src.config import get_settings
    settings = get_settings()
    print(f"Database URL: {settings.DATABASE_URL}")  # URL 확인용 로그
    return settings

@pytest_asyncio.fixture
async def db_manager(test_settings):
    # 테스트용 설정으로 DB 매니저 초기화
    print(f"Using Database URL: {test_settings.DATABASE_URL}")  # URL 확인용 로그
    db_manager = DatabaseManager(db_url=test_settings.DATABASE_URL)
    await db_manager.init_db()
    yield db_manager  # yield로 변경

@pytest_asyncio.fixture
async def test_user(db_manager):
    # db_manager는 이미 fixture에서 await되어 있으므로 다시 await하지 않음
    async with db_manager.async_session() as session:
        test_user = User(
            email="test@example.com",
            name="Test User",
            provider="test",
            provider_id="test123"
        )
        session.add(test_user)
        await session.commit()
        await session.refresh(test_user)
        yield test_user  # yield로 변경

@pytest.mark.asyncio
async def test_store_chat_messages(db_manager, test_user):
    try:
        # 테스트용 메시지 데이터 생성
        messages = [
            {
                "chat_id": "test_chat",
                "user_id": test_user.id,
                "sender": "Test User",
                "message": "Hello, World!",
                "datetime": datetime.utcnow(),
                "total_morphemes": 5,
                "substantives": 2,
                "predicates": 1
            },
            {
                "chat_id": "test_chat",
                "user_id": test_user.id,
                "sender": "Test User",
                "message": "How are you?",
                "datetime": datetime.utcnow(),
                "total_morphemes": 3,
                "substantives": 1,
                "predicates": 1
            }
        ]
        
        # 메시지 저장
        await db_manager.store_chat_messages(messages)
        
        # 저장된 메시지 확인
        async with db_manager.async_session() as session:
            query = select(ChatMessage).where(ChatMessage.chat_id == "test_chat")
            result = await session.execute(query)
            stored_messages = result.scalars().all()
            
            assert len(stored_messages) == 2
            assert stored_messages[0].message == "Hello, World!"
            assert stored_messages[0].user_id == test_user.id
    
    finally:
        # 테스트 후 cleanup - 메시지 먼저 삭제
        async with db_manager.async_session() as session:
            # 메시지 삭제
            await session.execute(
                delete(ChatMessage).where(ChatMessage.chat_id == "test_chat")
            )
            # 사용자 삭제
            await session.execute(
                delete(User).where(User.id == test_user.id)
            )
            await session.commit() 