# L.U.N.A. 메모리 시스템 가이드

## 개요

대화 컨텍스트를 저장하고 관리하여 연속적인 대화가 가능한 메모리 시스템입니다.

## 주요 기능

### 1. 자동 메모리 저장
- 모든 대화가 자동으로 저장됩니다
- 최대 50개의 대화 저장 (오래된 것부터 삭제)
- JSON 형식으로 `./memory/conversation_history.json`에 저장

### 2. 컨텍스트 관리
- LLM에 최근 6턴의 대화를 컨텍스트로 전달
- 대화 흐름을 이해하고 자연스러운 응답 생성

### 3. 메타데이터 저장
- 타임스탬프
- 감정 분석 결과
- 인텐트 분류
- LLM 모드 (API/서버)

## 설정

### 기본 설정

```python
memory_service = MemoryService(
    memory_dir="./memory",          # 저장 디렉토리
    max_entries=50,                 # 최대 저장 개수
    max_context_turns=6             # LLM에 전달할 최대 턴 수
)
```

### 커스터마이징

`main.py`에서 설정 변경:

```python
memory_service = MemoryService(
    memory_dir="./memory",
    max_entries=100,                # 더 많은 대화 저장
    max_context_turns=10            # 더 긴 컨텍스트 윈도우
)
```

## API 엔드포인트

### 1. 메모리 통계 조회

**GET** `/memory/stats`

응답 예시:
```json
{
  "total_entries": 25,
  "first_conversation": "2025-10-21T10:30:00",
  "last_conversation": "2025-10-21T14:45:00",
  "context_window": 6,
  "max_stored": 50
}
```

### 2. 최근 대화 조회

**GET** `/memory/recent?count=10`

응답 예시:
```json
{
  "recent_conversations": [
    {
      "timestamp": "2025-10-21T14:45:00",
      "user": "안녕하세요",
      "assistant": "음, 다엘. 내 생각 하고 있었어?",
      "metadata": {
        "emotion": "neutral",
        "intent": "general",
        "mode": "api"
      }
    }
  ]
}
```

### 3. 메모리 삭제

**DELETE** `/memory/clear`

응답:
```json
{
  "status": "success",
  "message": "메모리가 삭제되었습니다."
}
```

## 사용 예시

### Python으로 API 호출

```python
import requests

# 메모리 통계 확인
response = requests.get("http://localhost:8000/memory/stats")
print(response.json())

# 최근 5개 대화 조회
response = requests.get("http://localhost:8000/memory/recent?count=5")
print(response.json())

# 메모리 삭제
response = requests.delete("http://localhost:8000/memory/clear")
print(response.json())
```

### PowerShell로 API 호출

```powershell
# 메모리 통계 확인
curl http://localhost:8000/memory/stats

# 최근 대화 조회
curl http://localhost:8000/memory/recent?count=5

# 메모리 삭제
curl -X DELETE http://localhost:8000/memory/clear
```

## 저장 형식

`./memory/conversation_history.json`:

```json
[
  {
    "timestamp": "2025-10-21T14:45:00.123456",
    "user": "안녕하세요",
    "assistant": "음, 다엘. 내 생각 하고 있었어?",
    "metadata": {
      "emotion": "neutral",
      "intent": "general",
      "mode": "api"
    }
  },
  {
    "timestamp": "2025-10-21T14:46:15.234567",
    "user": "오늘 날씨 어때?",
    "assistant": "창 밖을 봐. 같이 확인하지 않을래?",
    "metadata": {
      "emotion": "curious",
      "intent": "general",
      "mode": "api"
    }
  }
]
```

## 동작 방식

### 대화 흐름

```
1. 사용자 입력
   ↓
2. 메모리에서 최근 6턴 로드
   ↓
3. 시스템 프롬프트 + 메모리 컨텍스트 + 현재 입력 → LLM
   ↓
4. LLM 응답 생성
   ↓
5. 메모리에 대화 저장 (사용자 입력 + 어시스턴트 응답)
   ↓
6. 응답 반환
```

### 컨텍스트 예시

```python
# LLM에 전달되는 메시지 구조
[
    {"role": "system", "content": "페르소나 프롬프트..."},
    {"role": "user", "content": "이전 대화 1"},
    {"role": "assistant", "content": "이전 응답 1"},
    {"role": "user", "content": "이전 대화 2"},
    {"role": "assistant", "content": "이전 응답 2"},
    # ... 최대 6턴
    {"role": "user", "content": "현재 사용자 입력"}
]
```

## 효율성 최적화

### 1. 제한된 저장 공간
- 최대 50개 대화만 저장
- 디스크 사용량 최소화

### 2. 제한된 컨텍스트 윈도우
- LLM에 최근 6턴만 전달
- 토큰 사용량 최적화
- 응답 속도 향상

### 3. 자동 정리
- `max_entries` 초과 시 오래된 대화 자동 삭제
- 메모리 관리 자동화

## 로그 확인

서버 실행 시:
```
[L.U.N.A. Startup] 메모리 서비스 초기화 중...
[L.U.N.A. Startup] 메모리 서비스 초기화 완료 (저장된 대화: 25개)
```

대화 처리 시:
```
[Memory] 대화 저장 완료 (총 26개)
```

## 주의사항

1. **파일 백업**: `./memory/` 디렉토리를 정기적으로 백업하세요
2. **개인정보**: 민감한 대화는 주기적으로 삭제하세요
3. **디스크 공간**: 장기간 사용 시 메모리 파일 크기 확인
4. **컨텍스트 길이**: 너무 긴 컨텍스트는 응답 품질 저하 가능

## 문제 해결

### 메모리 파일 손상 시
```python
# 메모리 삭제 후 재시작
DELETE /memory/clear
```

### 메모리 파일 위치 변경
`main.py` 수정:
```python
memory_service = MemoryService(
    memory_dir="./custom_memory_dir"
)
```

## 향후 개선 계획

- [ ] 대화 검색 기능
- [ ] 중요 대화 북마크
- [ ] 감정 기반 메모리 필터링
- [ ] 장기 메모리 요약 기능
