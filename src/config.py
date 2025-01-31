from pydantic_settings import BaseSettings
import os
from urllib.parse import quote_plus

class Settings(BaseSettings):
    # JWT 설정
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    
    # 데이터베이스 설정
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_NAME: str
    
    @property
    def DATABASE_URL(self) -> str:
        # URL 생성 전 값 검증
        if not all([self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_NAME]):
            raise ValueError("Database configuration is incomplete")
        
        # 특수문자가 포함된 비밀번호를 URL 인코딩
        encoded_password = quote_plus(self.DB_PASSWORD)
        return f"postgresql+asyncpg://{self.DB_USER}:{encoded_password}@{self.DB_HOST}/{self.DB_NAME}"
    
    # OAuth2 설정
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/google/callback"
    
    NAVER_CLIENT_ID: str
    NAVER_CLIENT_SECRET: str
    NAVER_REDIRECT_URI: str = "http://localhost:8000/auth/naver/callback"
    
    KAKAO_CLIENT_ID: str
    KAKAO_CLIENT_SECRET: str
    KAKAO_REDIRECT_URI: str = "http://localhost:8000/auth/kakao/callback"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

def get_settings():
    env_file = os.getenv("ENV_FILE", ".env")
    return Settings(_env_file=env_file)

settings = get_settings() 