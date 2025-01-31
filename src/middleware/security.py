from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from starlette.middleware.sessions import SessionMiddleware

def setup_security(app: FastAPI):
    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://your-frontend-domain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 신뢰할 수 있는 호스트 설정
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["your-api-domain.com"]
    )
    
    # 세션 미들웨어
    app.add_middleware(
        SessionMiddleware,
        secret_key="your-secret-key",
        same_site="lax",  # CSRF 보호
        https_only=True
    )

# Rate Limiting 구현
from fastapi import Request
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def check_rate_limit(self, client_ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        # 1분 이내의 요청만 유지
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        self.requests[client_ip].append(now)
        return True 