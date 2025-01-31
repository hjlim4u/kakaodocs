from pydantic import BaseModel, EmailStr
from typing import Optional

class User(BaseModel):
    id: int
    email: EmailStr
    name: str
    provider: str
    provider_id: str
    
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None
    
class OAuthUserInfo(BaseModel):
    provider: str
    provider_id: str
    email: EmailStr
    name: str 