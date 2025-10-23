# 메모리 시스템 가이드

L.U.N.A. 메모리 시스템은 SQLite 데이터베이스 기반으로 대화를 저장하고, 토큰 제한 컨텍스트 창을 통해 LLM에 효율적으로 과거 대화를 전달합니다.

## 목차

1. [개요](#개요)
2. [아키텍처](#아키텍처)
3. [설정](#설정)
4. [주요 기능](#주요-기능)
5. [사용 방법](#사용-방법)
6. [마이그레이션](#마이그레이션)
7. [API 레퍼런스](#api-레퍼런스)
8. [고급 기능](#고급-기능)

---

## 개요

### 주요 특징

- ✅ **SQLite 기반 저장소**: 빠르고 안정적인 로컬 데이터베이스
- ✅ **전문 검색(FTS5)**: 대화 내용 전체 텍스트 검색
- ✅ **토큰 제한 컨텍스트**: LLM 토큰 비용 절감 및 효율성 향상
- ✅ **자동 요약**: 오래된 대화를 LLM으로 요약하여 보관
- ✅ **RESTful API**: 프론트엔드 통합을 위한 HTTP 엔드포인트

### 기존 시스템과의 차이

| 항목 | 기존 (JSON 파일) | 현재 (SQLite) |
|------|------------------|---------------|
| 저장 방식 | JSON 파일 | SQLite DB |
| 검색 기능 | ❌ 없음 | ✅ FTS5 전문 검색 |
| 동시성 | ⚠️ 제한적 | ✅ 안전 |
| 토큰 관리 | ❌ 없음 | ✅ 토큰 예산 기반 |
| 확장성 | ⚠️ 낮음 | ✅ 높음 |

---

## 아키텍처

### 구조

```
services/memory/
├── __init__.py              # 호환성 래퍼
├── database.py              # SQLite 연결 관리
├── models.py                # Pydantic 데이터 모델
├── repository.py            # DB CRUD 작업
├── memory_service.py        # 비즈니스 로직
└── token_utils.py           # 토큰 추정 유틸리티
```

### 데이터 흐름

```
사용자 입력
    ↓
InteractionService
    ↓
MemoryService.add_entry()
    ↓
Repository.create_conversation()
    ↓
SQLite DB (conversations 테이블)
    ↓
MemoryService.get_context_for_llm()
    ↓
토큰 제한 컨텍스트 조립
    ↓
LLM에 전달
```

### 데이터베이스 스키마

#### conversations 테이블
```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    emotion TEXT,
    intent TEXT,
    processing_time REAL,
    cached BOOLEAN DEFAULT 0,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### summaries 테이블
```sql
CREATE TABLE summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    content TEXT NOT NULL,
    summarized_turns INTEGER,
    start_conversation_id INTEGER,
    end_conversation_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### FTS5 검색 테이블
```sql
CREATE VIRTUAL TABLE conversations_fts USING fts5(
    user_message, 
    assistant_message,
    content='conversations',
    content_rowid='id'
);
```

---

## 설정

### config/models.yaml

```yaml
memory:
  max_entries: 50              # 최대 저장 대화 수
  max_context_turns: 6         # LLM에 전달할 최근 대화 턴 수
  max_context_tokens: 1500     # LLM 컨텍스트 최대 토큰 수 (0이면 턴 기반만 사용)
  summary_threshold: 20        # 자동 요약 실행 기준 (턴 수)
  enable_auto_summary: true    # 자동 요약 활성화
  summary_language: "korean"   # 요약 언어 (korean/english)
```

### 설정 항목 설명

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `max_entries` | 50 | DB에 저장할 최대 대화 수 (현재는 제한 없음) |
| `max_context_turns` | 6 | LLM에 전달할 최대 대화 턴 수 |
| `max_context_tokens` | 1500 | LLM 컨텍스트 최대 토큰 수 (토큰 예산) |
| `summary_threshold` | 20 | 자동 요약이 트리거되는 대화 턴 수 |
| `enable_auto_summary` | true | 자동 요약 활성화 여부 |
| `summary_language` | korean | 요약 생성 언어 |

---

## 주요 기능

### 1. 토큰 제한 컨텍스트 창

**목적**: LLM에 전달되는 컨텍스트 토큰 수를 제한하여 비용 절감 및 효율성 향상

**동작 방식**:
1. 최신 요약을 시스템 메시지로 포함
2. 최근 대화부터 토큰 예산 내에서 추가
3. 예산 초과 시 오래된 대화 제외
4. 최소 1개의 최근 대화는 보장

**예제**:
```python
# config.yaml에서 설정
memory:
  max_context_tokens: 1500  # 최대 1500 토큰

# 자동으로 적용됨
context = memory_service.get_context_for_llm()
# context는 1500 토큰 이내로 조립됨
```

**토큰 추정 방식**:
- 간단한 휴리스틱: 평균 4자 = 1토큰
- 메시지당 오버헤드: 4토큰 (role, content 메타데이터)
- 보수적 추정으로 예산 초과 방지

### 2. 자동 요약

**목적**: 오래된 대화를 요약하여 컨텍스트 효율성 유지

**트리거 조건**:
- `summary_threshold` 대화 수 도달 시 (기본: 20턴)
- `enable_auto_summary: true` 설정 시

**동작 방식**:
1. 대화가 20턴 이상 누적되면 자동 실행
2. 최근 `max_context_turns`(6턴)는 유지
3. 나머지 오래된 대화를 LLM으로 요약
4. 요약본을 summaries 테이블에 저장
5. (선택) 원본 대화 삭제

**수동 요약**:
```python
# 수동으로 요약 실행
success = memory_service.force_summarize()
```

### 3. 전문 검색 (FTS5)

**목적**: 과거 대화 내용을 빠르게 검색

**검색 방법**:
```python
# 키워드로 대화 검색
results = memory_service.search_conversations(
    keyword="날씨",
    limit=10
)

for conv in results:
    print(f"사용자: {conv.user_message}")
    print(f"어시스턴트: {conv.assistant_message}")
```

**REST API**:
```bash
POST /memory/conversations/search
Content-Type: application/json

{
  "keyword": "날씨",
  "limit": 10
}
```

### 4. 메타데이터 저장

**저장 가능한 메타데이터**:
- `emotion`: 감정 분석 결과
- `intent`: 의도 분류 결과
- `processing_time`: 처리 시간 (초)
- `cached`: 캐시 사용 여부
- 커스텀 JSON 데이터

**예제**:
```python
memory_service.add_entry(
    user_input="안녕하세요",
    assistant_response="네, 안녕하세요!",
    metadata={
        "emotion": "joy",
        "intent": "greeting",
        "processing_time": 0.5,
        "cached": False,
        "custom_field": "custom_value"
    }
)
```

---

## 사용 방법

### 기본 사용

```python
from services.memory import MemoryService

# 초기화
memory_service = MemoryService(
    memory_dir="./memory",
    max_context_turns=6,
    max_context_tokens=1500,
    enable_auto_summary=True,
    llm_service=llm_service
)

# 대화 추가
memory_service.add_entry(
    user_input="오늘 날씨 어때?",
    assistant_response="오늘은 맑고 화창합니다!",
    metadata={"emotion": "neutral", "intent": "weather"}
)

# LLM용 컨텍스트 가져오기
context = memory_service.get_context_for_llm()
# context = [
#   {"role": "system", "content": "[이전 대화 요약]\n..."},
#   {"role": "user", "content": "안녕하세요"},
#   {"role": "assistant", "content": "네, 안녕하세요!"},
#   ...
# ]

# 최근 대화 요약 텍스트
summary_text = memory_service.get_recent_summary(count=5)

# 통계 조회
stats = memory_service.get_memory_stats()
print(f"총 대화: {stats['conversations']}개")
print(f"요약: {stats['summaries']}개")

# 메모리 삭제
memory_service.clear_memory()
```

### REST API 사용

#### 대화 추가
```python
# InteractionService에서 자동으로 처리됨
```

#### 대화 목록 조회
```bash
GET /memory/conversations?limit=10&offset=0
```

#### 특정 대화 조회
```bash
GET /memory/conversations/{conversation_id}
```

#### 대화 검색
```bash
POST /memory/conversations/search
Content-Type: application/json

{
  "keyword": "검색어",
  "limit": 10
}
```

#### 대화 삭제
```bash
DELETE /memory/conversations/{conversation_id}
```

#### 요약 목록 조회
```bash
GET /memory/summaries?limit=10
```

#### 통계 조회
```bash
GET /memory/stats
```

#### 메모리 전체 삭제
```bash
DELETE /memory/clear?confirm=true
```

---

## 마이그레이션

### JSON → SQLite 마이그레이션

기존 JSON 기반 메모리를 SQLite로 마이그레이션하는 스크립트가 제공됩니다.

```bash
# 마이그레이션 실행
python scripts/migrate_memory.py

# 검증 옵션 추가
python scripts/migrate_memory.py --verify
```

**마이그레이션 과정**:
1. `memory/conversation_history.json` 로드
2. 각 대화를 `conversations` 테이블에 삽입
3. 요약 데이터가 있으면 `summaries` 테이블에 삽입
4. FTS5 인덱스 자동 업데이트
5. 검증 및 통계 출력

**백업 권장**:
```bash
# 마이그레이션 전 백업
cp memory/conversation_history.json memory/conversation_history.json.backup
```

---

## API 레퍼런스

### MemoryService

#### `__init__()`
```python
MemoryService(
    memory_dir: str = "./memory",
    max_entries: int = 50,
    max_context_turns: int = 6,
    max_context_tokens: int = 1500,
    summary_threshold: int = 20,
    enable_auto_summary: bool = True,
    llm_service = None,
    user_id: str = "default",
    session_id: str = "default"
)
```

#### `add_entry()`
```python
memory_service.add_entry(
    user_input: str,
    assistant_response: str,
    metadata: Optional[Dict[str, Any]] = None
)
```

#### `get_context_for_llm()`
```python
context: List[Dict[str, str]] = memory_service.get_context_for_llm(
    include_system_prompt: bool = False
)
```

**반환 형식**:
```python
[
    {"role": "system", "content": "[이전 대화 요약]\n..."},  # 요약이 있을 때
    {"role": "user", "content": "사용자 메시지"},
    {"role": "assistant", "content": "어시스턴트 응답"},
    ...
]
```

#### `get_conversations()`
```python
conversations: List[ConversationResponse] = memory_service.get_conversations(
    limit: int = 50,
    offset: int = 0
)
```

#### `search_conversations()`
```python
results: List[ConversationResponse] = memory_service.search_conversations(
    keyword: str,
    limit: int = 50
)
```

#### `get_memory_stats()`
```python
stats: Dict[str, Any] = memory_service.get_memory_stats()
# {
#     "total_entries": 100,
#     "conversations": 95,
#     "summaries": 5,
#     "first_conversation": "2025-01-01T00:00:00",
#     "last_conversation": "2025-10-22T16:00:00",
#     "context_window": 6,
#     "max_stored": 50,
#     "auto_summary_enabled": True,
#     "summary_threshold": 20
# }
```

#### `clear_memory()`
```python
memory_service.clear_memory()
```

#### `force_summarize()`
```python
success: bool = memory_service.force_summarize()
```

---

## 고급 기능

### 1. 다중 사용자/세션 지원

```python
# 사용자별 메모리 서비스
user1_memory = MemoryService(
    memory_dir="./memory",
    user_id="user_001",
    session_id="session_001"
)

user2_memory = MemoryService(
    memory_dir="./memory",
    user_id="user_002",
    session_id="session_002"
)
```

### 2. 토큰 추정 커스터마이징

```python
# services/memory/token_utils.py
CHARS_PER_TOKEN = 4.0  # 기본값

# 더 정확한 추정을 위해 조정 가능
# - 한국어 중심: 3.5~4.0
# - 영어 중심: 4.0~4.5
# - 코드 중심: 3.0~3.5
```

### 3. 토큰 예산 비활성화

```yaml
# config/models.yaml
memory:
  max_context_tokens: 0  # 0으로 설정하면 턴 기반만 사용
```

### 4. 커스텀 요약 프롬프트

`MemoryService._generate_summary()` 메서드를 수정하여 요약 프롬프트를 커스터마이징할 수 있습니다.

### 5. 임베딩 기반 관련성 선택 (향후 기능)

현재는 최신성 기반으로 대화를 선택하지만, 향후 임베딩 벡터를 사용하여 현재 질문과 가장 관련 높은 과거 대화를 선택하는 기능을 추가할 예정입니다.

---

## 트러블슈팅

### DB 잠금 오류

**증상**: `database is locked` 오류

**해결책**:
- DatabaseManager는 스레드별 연결을 사용하므로 대부분의 경우 문제없음
- 여러 프로세스에서 동시 접근 시 발생 가능
- 필요시 WAL 모드 활성화:
  ```python
  # database.py에서
  conn.execute("PRAGMA journal_mode=WAL")
  ```

### 토큰 수 불일치

**증상**: 실제 LLM 사용 토큰과 추정치가 다름

**원인**: 휴리스틱 추정 방식의 한계

**해결책**:
- `token_utils.py`의 `CHARS_PER_TOKEN` 값 조정
- 또는 `tiktoken` 같은 정확한 토큰라이저 사용 (의존성 추가 필요)

### 메모리 사용량 증가

**증상**: DB 파일 크기 계속 증가

**해결책**:
- 주기적으로 오래된 대화 삭제
- VACUUM 명령으로 DB 최적화:
  ```python
  db_manager.get_connection().execute("VACUUM")
  ```

---

## 성능 최적화

### 인덱스

현재 제공되는 인덱스:
- `idx_conversations_user_session`: user_id, session_id, timestamp
- FTS5 전문 검색 인덱스

### 쿼리 최적화

- LIMIT/OFFSET 사용으로 대량 데이터 페이지네이션
- 필요한 컬럼만 선택 (SELECT *)
- 트랜잭션 배치 처리

---

## 참고 자료

- [SQLite 공식 문서](https://www.sqlite.org/docs.html)
- [FTS5 전문 검색](https://www.sqlite.org/fts5.html)
- [Pydantic 문서](https://docs.pydantic.dev/)

---

## 변경 이력

### v1.0.0 (2025-10-22)
- ✅ SQLite 기반 메모리 시스템 구현
- ✅ 토큰 제한 컨텍스트 창 추가
- ✅ FTS5 전문 검색 지원
- ✅ REST API 엔드포인트 추가
- ✅ JSON → SQLite 마이그레이션 스크립트
- ✅ 27개 단위 테스트 통과
