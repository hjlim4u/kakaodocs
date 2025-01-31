from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
import httpx
from typing import Optional
from sqlalchemy.orm import Session
import secrets
from sqlalchemy.future import select

from ..config import settings
from ..models.user import User, TokenResponse, OAuthUserInfo, UserSession
from ..utils.db_manager import DatabaseManager
from ..services.oauth_service import OAuthService
from ..utils.auth import create_tokens

router = APIRouter(prefix="/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def generate_refresh_token() -> str:
    return secrets.token_urlsafe(32)

@router.get("/google")
async def google_login():
    return {
        "url": f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={settings.GOOGLE_CLIENT_ID}&"
        f"response_type=code&"
        f"scope=email profile&"
        f"redirect_uri={settings.GOOGLE_REDIRECT_URI}"
    }

@router.get("/google/callback")
async def google_callback(code: str):
    oauth_service = OAuthService()
    oauth_user = await oauth_service.get_google_user_info(code)
    user = await oauth_service.save_or_update_user(oauth_user)
    
    tokens = await create_tokens(user.id)
    return TokenResponse(**tokens)

@router.get("/naver")
async def naver_login():
    return {
        "url": f"https://nid.naver.com/oauth2.0/authorize?"
        f"client_id={settings.NAVER_CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={settings.NAVER_REDIRECT_URI}"
    }

@router.get("/naver/callback")
async def naver_callback(code: str):
    oauth_service = OAuthService()
    oauth_user = await oauth_service.get_user_info("naver", code)
    user = await oauth_service.save_or_update_user(oauth_user)
    
    tokens = await create_tokens(user.id)
    return TokenResponse(**tokens)

@router.get("/kakao")
async def kakao_login():
    return {
        "url": f"https://kauth.kakao.com/oauth/authorize?"
        f"client_id={settings.KAKAO_CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={settings.KAKAO_REDIRECT_URI}"
    }

@router.get("/kakao/callback")
async def kakao_callback(code: str):
    oauth_service = OAuthService()
    oauth_user = await oauth_service.get_user_info("kakao", code)
    user = await oauth_service.save_or_update_user(oauth_user)
    
    tokens = await create_tokens(user.id)
    return TokenResponse(**tokens)

@router.post("/refresh")
async def refresh_token(refresh_token: str):
    """리프레시 토큰을 사용하여 새로운 액세스 토큰 발급"""
    db_manager = DatabaseManager()
    
    # 리프레시 토큰 검증
    session_data = await db_manager.verify_refresh_token(refresh_token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # 새로운 액세스 토큰 생성
    access_token = await create_access_token({"sub": session_data["email"]})
    
    # 새로운 리프레시 토큰 생성 및 업데이트
    new_refresh_token = generate_refresh_token()
    expires_at = datetime.utcnow() + timedelta(days=30)
    
    success = await db_manager.update_refresh_token(
        session_data["session_id"],
        new_refresh_token,
        expires_at
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update refresh token"
        )
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }
