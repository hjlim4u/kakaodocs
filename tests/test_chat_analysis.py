import pytest
import pytest_asyncio
import os
from datetime import datetime
from pathlib import Path
from src.utils.db_manager import DatabaseManager
from src.database.models import User, ChatMessage
from src.utils.chat_analyzer import ChatAnalyzer
from sqlalchemy import select, delete
from src.config import get_settings
from src.utils.cache_manager import CacheManager
from src.utils.text_processor import TextProcessor
from src.utils.interaction_analyzer import InteractionAnalyzer

# 테스트용 설정 로드
@pytest.fixture
def test_settings():
    os.environ['ENV_FILE'] = '.env.test'
    settings = get_settings()
    print(f"Database URL: {settings.DATABASE_URL}")
    return settings

@pytest_asyncio.fixture
async def db_manager(test_settings):
    db_manager = DatabaseManager(db_url=test_settings.DATABASE_URL)
    await db_manager.init_db()
    yield db_manager

@pytest_asyncio.fixture
async def test_user(db_manager):
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
        yield test_user

@pytest.fixture
def analyzer(db_manager):
    # cache_manager = CacheManager()
    # text_processor = TextProcessor()
    # interaction_analyzer = InteractionAnalyzer(text_processor, db_manager)
    return ChatAnalyzer()

@pytest.mark.asyncio
async def test_analyze_chat_with_user(db_manager, test_user, analyzer):
    try:
        # 테스트용 채팅 파일 경로
        chat_file_path = Path("Talk_2025.1.15 19_40-1_ios.txt")
        assert chat_file_path.exists(), "Chat file not found"
        
        # analyzer fixture 사용 (직접 생성하지 않음)
        result = await analyzer.analyze_single_chat(
            str(chat_file_path),
            user_id=test_user.id,
            is_authenticated=True
        )
        
        # 분석 결과 검증
        assert result is not None
        assert 'chat_id' in result
        assert 'basic_stats' in result
        assert 'sentiment_analysis' in result  # 감성 분석 결과 확인
        
        # 저장된 메시지 확인
        async with db_manager.async_session() as session:
            query = select(ChatMessage).where(ChatMessage.user_id == test_user.id)
            result = await session.execute(query)
            stored_messages = result.scalars().all()
            
            assert len(stored_messages) > 0
            
    finally:
        # 테스트 후 cleanup
        async with db_manager.async_session() as session:
            # 메시지 먼저 삭제
            await session.execute(
                delete(ChatMessage).where(ChatMessage.user_id == test_user.id)
            )
            # 사용자 삭제
            await session.execute(
                delete(User).where(User.id == test_user.id)
            )
            await session.commit() 