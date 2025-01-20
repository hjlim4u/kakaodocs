여기 프로젝트를 위한 README.md를 작성해드리겠습니다:

```markdown
# KakaoTalk Chat Analyzer

카카오톡 채팅방 대화 내용을 분석하여 대화 패턴과 통계를 추출하는 Python 기반 분석 도구입니다.

## 주요 기능

- 기본 통계 분석
  - 총 메시지 수
  - 참여자 수 
  - 평균 메시지 길이
  - 메시지 길이 표준편차
  - 참여자별 메시지 수

- 대화 구간 분석
  - 응답 시간 패턴
  - 시간대별 활동량
  - 참여 불균형도
  - 중심성 분석

- 언어 스타일 분석
  - 어휘 다양성
  - 이모지 사용률
  - 형태소 분석
  - 품사별 사용 비율

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/kakaodocs.git
cd kakaodocs
```

2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 카카오톡 채팅방 내보내기 파일을 프로젝트 루트 디렉토리에 위치시킵니다.

2. main.py 실행:
```bash
python src/main.py
```

3. 분석 결과는 콘솔에 출력되며, JSON 파일로도 저장됩니다.

## 요구사항

- Python 3.12+
- FastAPI
- pandas
- pydantic
- PyKoSpacing
- KoNLPy
- networkx
- scipy
- scikit-learn
- transformers
- torch

## 테스트

```bash
pytest
```

## 라이선스

MIT License
