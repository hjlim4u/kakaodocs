from typing import Optional
import httpx
from ..models.user import OAuthUserInfo, User
from ..config import settings
from ..utils.db_manager import DatabaseManager
from sqlalchemy.future import select

class OAuthService:
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    async def get_user_info(self, provider: str, code: str) -> OAuthUserInfo:
        if provider == "google":
            return await self._get_google_user_info(code)
        elif provider == "naver":
            return await self._get_naver_user_info(code)
        elif provider == "kakao":
            return await self._get_kakao_user_info(code)
        raise ValueError(f"Unsupported provider: {provider}")
    
    async def save_or_update_user(self, oauth_user: OAuthUserInfo) -> User:
        async with self.db_manager.async_session() as session:
            query = select(User).where(
                User.email == oauth_user.email,
                User.provider == oauth_user.provider
            )
            result = await session.execute(query)
            user = result.scalar_one_or_none()
            
            if user:
                user.name = oauth_user.name
                user.provider_id = oauth_user.provider_id
            else:
                user = User(
                    email=oauth_user.email,
                    name=oauth_user.name,
                    provider=oauth_user.provider,
                    provider_id=oauth_user.provider_id
                )
                session.add(user)
            
            await session.commit()
            await session.refresh(user)
            return user

    async def _get_google_user_info(self, code: str) -> OAuthUserInfo:
        async with httpx.AsyncClient() as client:
            token = await self._get_google_token(client, code)
            user_data = await self._get_google_user_data(client, token)
            return OAuthUserInfo(
                provider="google",
                provider_id=user_data["id"],
                email=user_data["email"],
                name=user_data["name"]
            )

    async def _get_google_token(self, client: httpx.AsyncClient, code: str) -> str:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.GOOGLE_REDIRECT_URI
            }
        )
        return response.json()["access_token"]

    async def _get_google_user_data(self, client: httpx.AsyncClient, token: str) -> dict:
        response = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()

    async def _get_naver_user_info(self, code: str) -> OAuthUserInfo:
        async with httpx.AsyncClient() as client:
            # 액세스 토큰 획득
            token = await self._get_naver_token(client, code)
            # 사용자 정보 획득
            user_data = await self._get_naver_user_data(client, token)
            
            return OAuthUserInfo(
                provider="naver",
                provider_id=user_data["id"],
                email=user_data["email"],
                name=user_data["name"]
            )

    async def _get_naver_token(self, client: httpx.AsyncClient, code: str) -> str:
        response = await client.post(
            "https://nid.naver.com/oauth2.0/token",
            data={
                "client_id": settings.NAVER_CLIENT_ID,
                "client_secret": settings.NAVER_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.NAVER_REDIRECT_URI
            }
        )
        return response.json()["access_token"]

    async def _get_naver_user_data(self, client: httpx.AsyncClient, token: str) -> dict:
        response = await client.get(
            "https://openapi.naver.com/v1/nid/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()["response"]

    async def _get_kakao_user_info(self, code: str) -> OAuthUserInfo:
        async with httpx.AsyncClient() as client:
            # 액세스 토큰 획득
            token = await self._get_kakao_token(client, code)
            # 사용자 정보 획득
            user_data = await self._get_kakao_user_data(client, token)
            
            return OAuthUserInfo(
                provider="kakao",
                provider_id=str(user_data["id"]),
                email=user_data["kakao_account"]["email"],
                name=user_data["properties"]["nickname"]
            )

    async def _get_kakao_token(self, client: httpx.AsyncClient, code: str) -> str:
        response = await client.post(
            "https://kauth.kakao.com/oauth/token",
            data={
                "client_id": settings.KAKAO_CLIENT_ID,
                "client_secret": settings.KAKAO_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.KAKAO_REDIRECT_URI
            }
        )
        return response.json()["access_token"]

    async def _get_kakao_user_data(self, client: httpx.AsyncClient, token: str) -> dict:
        response = await client.get(
            "https://kapi.kakao.com/v2/user/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
