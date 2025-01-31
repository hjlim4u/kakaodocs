# Python 3.9 이미지를 기반으로 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (KoNLPy 의존성)
RUN apt-get update && \
    apt-get install -y g++ default-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/

# 실행 명령
CMD ["python", "src/main.py"] 