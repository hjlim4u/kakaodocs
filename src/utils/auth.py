from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from .db_manager import DatabaseManager, User, UserSession
from sqlalchemy.future import select
from ..config import settings

class TokenError(Exception):
    """토큰 관련 예외"""
    pass

async def create_jwt_token(data: dict, expires_delta: timedelta) -> str:
    """JWT 토큰 생성"""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

async def save_refresh_token(user_id: int, refresh_token: str, expires_delta: timedelta):
    """리프레시 토큰을 데이터베이스에 저장"""
    db_manager = DatabaseManager()
    expires_at = datetime.utcnow() + expires_delta
    
    async with db_manager.async_session() as session:
        user_session = UserSession(
            user_id=user_id,
            refresh_token=refresh_token,
            expires_at=expires_at
        )
        session.add(user_session)
        await session.commit()

async def verify_refresh_token_in_db(user_id: str, refresh_token: str) -> bool:
    """데이터베이스에서 리프레시 토큰 검증"""
    db_manager = DatabaseManager()
    session_data = await db_manager.verify_refresh_token(refresh_token)
    return session_data is not None and str(session_data["user_id"]) == user_id

async def create_tokens(user_id: int):
    # Access Token 생성
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_jwt_token(
        data={"sub": str(user_id)},
        expires_delta=access_token_expires
    )
    
    # Refresh Token 생성
    refresh_token_expires = timedelta(days=30)
    refresh_token = create_jwt_token(
        data={"sub": str(user_id), "type": "refresh"},
        expires_delta=refresh_token_expires
    )
    
    # Refresh Token DB 저장
    await save_refresh_token(user_id, refresh_token, refresh_token_expires)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

async def refresh_access_token(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "refresh":
            raise TokenError()
            
        user_id = payload.get("sub")
        if not await verify_refresh_token_in_db(user_id, refresh_token):
            raise TokenError()
            
        return await create_tokens(int(user_id))
    except JWTError:
        raise TokenError()

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[dict]:
    """현재 사용자 정보를 가져옵니다."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 데이터베이스에서 사용자 정보 조회
        db_manager = DatabaseManager()
        async with db_manager.async_session() as session:
            query = select(User).where(User.email == email)
            result = await session.execute(query)
            user = result.scalar_one_or_none()
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return {
                "id": user.id,
                "email": user.email,
                "name": user.name
            }
            
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )