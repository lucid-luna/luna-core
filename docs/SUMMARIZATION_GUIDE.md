# L.U.N.A. 대화 요약 시스템 가이드

## 📚 개요

대화 요약 시스템은 오래된 대화를 자동으로 압축하여 LLM의 컨텍스트 윈도우를 효율적으로 관리하고, API 비용을 크게 절감합니다.

### ✨ 주요 특징

- **자동 요약**: 설정한 턴 수 도달 시 자동 실행
- **LLM 기반 요약**: Gemini API를 사용한 고품질 요약
- **컨텍스트 보존**: 중요한 정보는 유지하면서 토큰 절약
- **비용 절감**: 최대 60% API 비용 감소
- **메모리 효율**: 오래된 대화 압축으로 저장 공간 절약

---

## 🎯 작동 원리

### 기본 구조

```
[대화 히스토리]
├─ 요약 항목 (0-19턴 압축)
└─ 최근 대화 (20-25턴 원본 유지)

[LLM 컨텍스트]
├─ [System] 이전 대화 요약
├─ [User] 질문 20
├─ [Assistant] 답변 20
├─ [User] 질문 21
└─ ...
```

### 요약 트리거

```python
대화 턴 수 >= summary_threshold (기본: 20턴)
    ↓
자동으로 요약 실행
    ↓
오래된 대화 → 요약 (LLM 생성)
최근 N턴 → 원본 유지 (기본: 6턴)
```

### 요약 예시

**원본 대화 (15턴, ~1500 토큰)**
```
User: 파이썬 배우고 싶어요
Assistant: 파이썬은 초보자에게 좋은 언어입니다...
User: 변수 선언은?
Assistant: x = 10 처럼 하면 됩니다...
[... 13개 턴 더 ...]
```

**요약 결과 (~200 토큰)**
```
[이전 대화 요약]
사용자는 파이썬 프로그래밍을 처음 시작하며, 변수 선언, 함수 정의,
리스트 사용법, 반복문 등 기본 문법을 순차적으로 학습했습니다.
이후 클래스와 상속, 예외 처리, 파일 입출력까지 학습을 진행했으며,
현재는 pandas를 활용한 데이터 분석에 관심을 보이고 있습니다.
```

**절감 효과**: 1500 토큰 → 200 토큰 (87% 절감)

---

## 🚀 사용 방법

### 1. 설정 (`config/models.yaml`)

```yaml
memory:
  max_entries: 50              # 최대 저장 대화 수
  max_context_turns: 6         # LLM에 전달할 최근 대화 턴 수
  summary_threshold: 20        # 자동 요약 실행 기준 (턴 수)
  enable_auto_summary: true    # 자동 요약 활성화
  summary_language: "korean"   # 요약 언어
```

#### 설정 설명

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `max_entries` | 50 | 저장할 최대 대화 수 (요약 포함) |
| `max_context_turns` | 6 | 원본으로 유지할 최근 대화 수 |
| `summary_threshold` | 20 | N턴 도달 시 자동 요약 실행 |
| `enable_auto_summary` | true | 자동 요약 on/off |
| `summary_language` | korean | 요약 언어 (향후 지원) |

### 2. 자동 요약 (권장)

설정만 하면 자동으로 작동합니다!

```python
# main.py에서 자동으로 초기화됨
memory_service = MemoryService(
    memory_dir="./memory",
    max_entries=50,
    max_context_turns=6,
    summary_threshold=20,
    enable_auto_summary=True,
    llm_service=llm_service
)

# 대화 추가 시 자동으로 체크
memory_service.add_entry("질문", "답변")
# 20턴 도달 → 자동 요약 실행!
```

### 3. 수동 요약

필요할 때 수동으로 실행할 수도 있습니다.

#### Python API

```python
from services.memory import MemoryService

memory = MemoryService(...)
memory.force_summarize()  # 즉시 요약 실행

# 요약 확인
summary = memory.get_summary()
print(summary)
```

#### HTTP API

```bash
# 요약 실행
curl -X POST http://localhost:8000/memory/summarize

# 요약 조회
curl http://localhost:8000/memory/summary

# 메모리 통계
curl http://localhost:8000/memory/stats
```

#### 응답 예시

```json
{
  "status": "success",
  "message": "대화 요약이 완료되었습니다.",
  "summary": "사용자는 파이썬 프로그래밍을 처음 시작하며..."
}
```

---

## 📊 API 엔드포인트

### 1. `POST /memory/summarize`

**설명**: 수동으로 대화 요약 실행

**요청**:
```bash
curl -X POST http://localhost:8000/memory/summarize
```

**응답**:
```json
{
  "status": "success",
  "message": "대화 요약이 완료되었습니다.",
  "summary": "사용자는 파이썬 학습 중..."
}
```

### 2. `GET /memory/summary`

**설명**: 현재 저장된 요약 조회

**요청**:
```bash
curl http://localhost:8000/memory/summary
```

**응답**:
```json
{
  "summary": "사용자는 파이썬 학습 중..."
}
```

### 3. `GET /memory/stats`

**설명**: 메모리 통계 정보

**요청**:
```bash
curl http://localhost:8000/memory/stats
```

**응답**:
```json
{
  "total_entries": 25,
  "conversations": 24,
  "summaries": 1,
  "auto_summary_enabled": true,
  "summary_threshold": 20,
  "context_window": 6
}
```

---

## 💰 비용 절감 효과

### 시나리오: 100회 대화 (각 20턴)

#### 요약 없이
```
20턴 × 100토큰/턴 = 2,000 토큰/대화
100회 대화 = 200,000 토큰

비용 (Gemini 2.5 Flash): $0.015
```

#### 요약 사용
```
요약 생성: 700 토큰 (1회)
이후 매 대화: 200토큰(요약) + 600토큰(6턴) = 800 토큰
100회 대화 = 700 + (99 × 800) = 79,900 토큰

비용 (Gemini 2.5 Flash): $0.006
```

#### 절감 효과
- **토큰 절감**: 200,000 → 79,900 (60% 감소)
- **비용 절감**: $0.015 → $0.006 (60% 감소)
- **절감액**: $0.009 (100회 기준)

### 연간 비용 비교

월 10,000회 대화 기준:

| 항목 | 요약 없음 | 요약 사용 | 절감액 |
|------|----------|----------|--------|
| 월간 | $1.50 | $0.60 | **$0.90** |
| 연간 | $18.00 | $7.20 | **$10.80** |

---

## 🔄 요약 프로세스

### 1. 자동 요약 플로우

```
[대화 추가]
    ↓
대화 수 체크
    ↓
20턴 이상? → Yes
    ↓
오래된 대화 추출 (0-13턴)
    ↓
LLM에 요약 요청
    ↓
요약 생성 (~2초)
    ↓
메모리 재구성
[요약] + [최근 6턴]
    ↓
파일 저장
```

### 2. 요약 프롬프트

```python
"""
다음 대화 내역을 요약해주세요:

User: 질문 1
Assistant: 답변 1
[... 더 많은 대화 ...]

요약에는 다음 내용을 포함하세요:
- 사용자의 주요 관심사와 목적
- 중요한 결정 사항이나 합의 내용
- 반복되는 주제나 패턴
- 향후 대화에 참조가 필요한 정보

간결하고 핵심적인 내용만 포함하여 3-5문장으로 작성해주세요.
"""
```

### 3. 증분 요약 (기존 요약 업데이트)

```python
"""
기존 요약:
사용자는 파이썬 기본 문법을 학습했습니다...

추가 대화 내역:
User: pandas 사용법 알려주세요
Assistant: pandas는 데이터 분석 라이브러리입니다...

위의 기존 요약에 추가 대화 내용을 반영하여 업데이트된 요약을 작성해주세요.
"""
```

---

## 🛠️ 고급 사용법

### 1. 컨텍스트 확인

```python
memory = MemoryService(...)

# LLM에 전달될 전체 컨텍스트
context = memory.get_context_for_llm()

for msg in context:
    print(f"[{msg['role']}] {msg['content']}")

# 출력:
# [system] [이전 대화 요약] 사용자는...
# [user] 최근 질문 1
# [assistant] 최근 답변 1
# ...
```

### 2. 요약 상태 확인

```python
# 현재 요약 존재 여부
summary = memory.get_summary()
if summary:
    print("요약 있음:", summary)
else:
    print("요약 없음")

# 통계 정보
stats = memory.get_memory_stats()
print(f"요약 수: {stats['summaries']}")
print(f"대화 수: {stats['conversations']}")
```

### 3. 임계값 동적 조정

```python
# 짧은 대화 → 빠른 요약
memory.summary_threshold = 10

# 긴 대화 → 느린 요약
memory.summary_threshold = 30

# 자동 요약 비활성화
memory.enable_auto_summary = False
```

### 4. 메모리 내용 확인

```python
# 전체 메모리 로드
all_memory = memory.load_memory()

for entry in all_memory:
    if entry['type'] == 'summary':
        print(f"[요약] {entry['content'][:50]}...")
    else:
        print(f"[대화] {entry['user'][:30]}...")
```

---

## 📈 성능 최적화

### 1. 적절한 임계값 설정

```python
# 짧은 세션 (1-2회 대화)
summary_threshold = 30  # 거의 요약 안 함

# 중간 세션 (5-10회 대화)
summary_threshold = 20  # 기본값

# 긴 세션 (수십 회 대화)
summary_threshold = 15  # 자주 요약
```

### 2. 컨텍스트 윈도우 조정

```python
# 짧은 컨텍스트 (빠른 응답)
max_context_turns = 4

# 중간 컨텍스트 (균형)
max_context_turns = 6  # 기본값

# 긴 컨텍스트 (정확한 맥락)
max_context_turns = 10
```

### 3. 요약 타이밍

```python
# 대화 중간에 백그라운드로 실행 (비동기)
# 향후 지원 예정

# 현재는 동기 실행 (~2초)
# 사용자 체감: 20턴마다 1번만 발생
```

---

## 🐛 문제 해결

### 문제: 요약이 생성되지 않음

**원인**:
- LLM 서비스가 초기화되지 않음
- 대화 수가 임계값 미달

**해결**:
```python
# 1. LLM 서비스 확인
print(memory.llm_service)  # None이면 문제

# 2. 대화 수 확인
stats = memory.get_memory_stats()
print(f"대화 수: {stats['conversations']}")
print(f"임계값: {stats['summary_threshold']}")

# 3. 수동 실행
memory.force_summarize()
```

### 문제: 요약 품질이 낮음

**원인**:
- 대화가 너무 짧음
- 중요 정보 부족

**해결**:
```python
# 임계값 높이기 (더 많은 대화 후 요약)
summary_threshold = 30

# 또는 수동으로 적절한 시점에 실행
if is_important_conversation_end():
    memory.force_summarize()
```

### 문제: API 비용이 여전히 높음

**확인 사항**:
```python
# 1. 요약이 실제로 생성되는지 확인
summary = memory.get_summary()
print("요약:", summary)

# 2. 컨텍스트 크기 확인
context = memory.get_context_for_llm()
print(f"컨텍스트 메시지 수: {len(context)}")

# 3. 임계값이 너무 높지 않은지 확인
print(f"임계값: {memory.summary_threshold}")
```

---

## 📝 베스트 프랙티스

### 1. 자동 요약 활용

```python
# ✅ 권장: 자동 요약 활성화
enable_auto_summary = True

# ❌ 비권장: 수동 관리
enable_auto_summary = False
```

### 2. 적절한 컨텍스트 윈도우

```python
# ✅ 권장: 6-10턴 유지
max_context_turns = 6

# ❌ 비권장: 너무 작거나 큼
max_context_turns = 2   # 너무 작음
max_context_turns = 20  # 너무 큼 (요약 의미 없음)
```

### 3. 임계값 설정

```python
# ✅ 권장: 15-25턴
summary_threshold = 20

# ❌ 비권장
summary_threshold = 5   # 너무 자주 요약
summary_threshold = 50  # 너무 늦게 요약
```

### 4. 요약 확인

```python
# ✅ 권장: 주기적 확인
stats = memory.get_memory_stats()
if stats['summaries'] > 0:
    summary = memory.get_summary()
    print("현재 요약:", summary)
```

---

## 🔮 향후 계획

- [ ] **비동기 요약**: 백그라운드 요약 (사용자 대기 없음)
- [ ] **다국어 요약**: 한국어/영어 자동 감지
- [ ] **요약 레벨**: 간단/상세 요약 선택
- [ ] **요약 히스토리**: 이전 요약 버전 관리
- [ ] **스마트 임계값**: 대화 패턴에 따른 자동 조정
- [ ] **요약 품질 평가**: 자동 품질 체크

---

## 📚 관련 문서

- [LLM Manager 가이드](./LLM_MANAGER_GUIDE.md)
- [메모리 시스템 가이드](./MEMORY_GUIDE.md)
- [캐시 시스템 가이드](./CACHE_SYSTEM_GUIDE.md)
- [스트리밍 응답 가이드](./STREAMING_GUIDE.md)

---

## 📞 예제 코드

전체 예제는 `examples/summarization_example.py`를 참고하세요:

```bash
cd examples
python summarization_example.py
```

---

**업데이트**: 2025-10-21  
**버전**: 2.0.0  
**작성자**: Dael
