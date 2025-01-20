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

## 분석 결과 예시

### 기본 통계
- 총 메시지 수, 참여자 수, 평균 메시지 길이 등 기본 통계
- 참여자별 메시지 수 분포

### 대화 구간 분석
- 시간대별 활동량 그래프
- 응답 시간 패턴 분석
- 참여 불균형도 측정
- 대화 흐름의 중심성 분석

### 감정 분석
- 사용자별 감정 상태 분포
- 시간에 따른 감정 변화 추이
- 전체 대화의 감정 기조 분석

## 프로젝트 구조

```bash
kakaodocs/
├── src/
│   ├── utils/
│   │   ├── chat_analyzer.py      # 메인 분석 로직
│   │   ├── interaction_analyzer.py # 상호작용 분석
│   │   ├── pattern_analyzer.py   # 패턴 분석
│   │   ├── sentiment_analyzer.py # 감정 분석
│   │   ├── topic_analyzer.py     # 토픽 분석
│   │   └── text_processor.py     # 텍스트 처리
│   └── main.py
├── tests/
│   └── utils/
├── requirements.txt
└── README.md
```

## 개발 가이드

### 테스트 작성

새로운 기능을 추가할 때는 반드시 테스트를 작성해주세요:

```python
@pytest.mark.asyncio
async def test_new_feature():
    # 테스트 코드 작성
    assert result == expected
```

### 캐싱 활용

분석 결과는 자동으로 캐시됩니다. 캐시를 활용하여 반복적인 분석을 피할 수 있습니다.

## 주의사항

1. 개인정보 보호
   - 분석 결과에는 민감한 개인정보가 포함될 수 있으므로 주의가 필요합니다
   - 결과 파일 관리에 유의해주세요

2. 리소스 사용
   - 대용량 채팅 분석 시 메모리 사용량에 주의가 필요합니다
   - 긴 시간의 채팅 로그는 분할 처리를 권장합니다

## 기여 방법

1. 이슈 등록
   - 버그 리포트
   - 새로운 기능 제안
   - 문서 개선 제안

2. 풀 리퀘스트
   - 코드 컨벤션을 준수해주세요
   - 테스트 코드를 포함해주세요
   - 관련 문서를 업데이트해주세요

## 라이선스

MIT License

## 연락처

버그 리포트 및 기능 제안은 이슈 트래커를 이용해주세요.
