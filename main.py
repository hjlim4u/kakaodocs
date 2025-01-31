from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.routers import chat_analysis
from src.utils.db_manager import DatabaseManager
from src.middleware.security import setup_security

# 데이터베이스 매니저 초기화
db_manager = DatabaseManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 데이터베이스 초기화
    await db_manager.init_db()
    yield
    # Shutdown: cleanup if needed
    
app = FastAPI(
    title="Chat Analysis API",
    description="카카오톡 채팅 분석 API",
    version="1.0.0",
    lifespan=lifespan
)

# 보안 설정 적용
setup_security(app)

# 라우터 등록
app.include_router(chat_analysis.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
