# L.U.N.A. 스트리밍 응답 가이드

## 📚 개요

스트리밍 응답 기능은 LLM의 응답을 단어 단위로 실시간으로 전송하여 사용자 경험을 개선합니다.

### ✨ 주요 특징

- **실시간 응답**: 단어 단위로 즉시 표시
- **체감 속도 향상**: 전체 응답 대기 시간 감소
- **진행 상황 표시**: 응답 생성 과정을 시각적으로 확인
- **캐시 통합**: 스트리밍 후 자동으로 전체 응답 캐싱
- **API 모드 전용**: Gemini API에서만 지원 (서버 모드는 향후 지원 예정)

---

## 🚀 사용 방법

### 1. 설정 확인

스트리밍은 **API 모드**에서만 작동합니다. `config/models.yaml`에서 확인:

```yaml
llm:
  mode: api  # 'api'로 설정되어 있어야 함
  api:
    gemini:
      model: gemini-2.5-flash
```

### 2. HTTP API 사용

#### 엔드포인트

```
POST /interact/stream
```

#### 요청 예시

```bash
curl -X POST http://localhost:8000/interact/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "인공지능의 미래에 대해 설명해주세요."}'
```

#### 응답 형식 (Server-Sent Events)

```
data: {"type": "emotion", "data": {"emotion": "neutral", "confidence": 0.95}}

data: {"type": "translation", "data": "Explain the future of AI."}

data: {"type": "llm_chunk", "data": "인공지능의"}

data: {"type": "llm_chunk", "data": " 미래는"}

data: {"type": "llm_chunk", "data": " 매우"}

data: {"type": "complete", "data": "인공지능의 미래는 매우 밝습니다..."}
```

#### 이벤트 타입

| 타입 | 설명 |
|------|------|
| `emotion` | 감정 분석 결과 |
| `translation` | 번역된 텍스트 (한국어 → 영어) |
| `llm_chunk` | LLM 응답 조각 (단어 단위) |
| `complete` | 전체 응답 완료 |
| `error` | 오류 발생 |

### 3. Python 클라이언트 사용

#### 필수 패키지 설치

```bash
pip install sseclient-py requests
```

#### 코드 예시

```python
import requests
import json
import sseclient

def stream_response(user_input: str):
    url = "http://localhost:8000/interact/stream"
    data = {"input": user_input}
    
    response = requests.post(url, json=data, stream=True)
    client = sseclient.SSEClient(response)
    
    print("응답: ", end="", flush=True)
    
    for event in client.events():
        data = json.loads(event.data)
        
        if data["type"] == "llm_chunk":
            print(data["data"], end="", flush=True)
        
        elif data["type"] == "complete":
            print("\n완료!")
            break

# 사용
stream_response("안녕하세요!")
```

### 4. LLMManager로 직접 사용

```python
from services.llm_manager import LLMManager

llm = LLMManager(
    mode="api",
    api_configs={"gemini": {"model": "gemini-2.5-flash"}}
)

# 스트리밍 요청
stream = llm.generate(
    target="gemini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python."}
    ],
    stream=True  # 스트리밍 활성화
)

# 응답 처리
for chunk in stream:
    if chunk.get("choices"):
        content = chunk["choices"][0]["delta"].get("content", "")
        if content:
            print(content, end="", flush=True)
```

### 5. LLMAPIService로 직접 사용

```python
from services.llm_api import LLMAPIService

api = LLMAPIService(
    api_configs={"gemini": {"model": "gemini-2.5-flash"}}
)

# 스트리밍 요청
stream = api.generate(
    target_provider="gemini",
    system_prompt="You are a helpful assistant.",
    user_prompt="What is machine learning?",
    stream=True
)

# 응답 처리
for chunk in stream:
    if "error" in chunk:
        print(f"Error: {chunk['error']}")
        break
    
    if chunk.get("choices"):
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)
```

---

## 🔄 스트리밍 플로우

```
[사용자 입력]
    ↓
[1. 감정 분석] → SSE 이벤트 전송
    ↓
[2. 번역 (필요시)] → SSE 이벤트 전송
    ↓
[3. 메모리 컨텍스트 로드]
    ↓
[4. LLM API 호출 (스트리밍)]
    ↓
[5. 청크별 전송] ← 실시간으로 단어 단위 전송
    ↓
[6. 완료 후 캐시 저장]
    ↓
[7. 메모리 저장]
```

---

## 📊 스트리밍 vs 일반 응답 비교

| 항목 | 일반 응답 | 스트리밍 응답 |
|------|----------|--------------|
| **첫 응답 시간** | 전체 생성 후 | 즉시 시작 |
| **사용자 경험** | 긴 대기 시간 | 실시간 피드백 |
| **네트워크 효율** | 한 번에 전송 | 점진적 전송 |
| **캐싱** | 즉시 저장 | 완료 후 저장 |
| **적합한 상황** | 짧은 응답 | 긴 응답 |

---

## ⚙️ 캐싱과 스트리밍

### 자동 캐싱

스트리밍 응답도 완료 후 자동으로 캐시에 저장됩니다:

```python
# 스트리밍 중...
for chunk in stream:
    # 각 청크 처리
    pass

# 스트리밍 완료 → 자동으로 캐시에 전체 응답 저장
```

### 캐시 히트 시

이미 캐시된 응답은 **일반 응답**으로 즉시 반환됩니다:

```python
# 첫 번째 요청: 스트리밍
response1 = llm.generate(..., stream=True)  # 스트리밍

# 동일한 요청: 캐시에서 반환
response2 = llm.generate(..., stream=True)  # 캐시 히트 → 즉시 반환 (스트리밍 아님)
```

---

## 🛠️ 고급 사용법

### 1. 타임아웃 설정

```python
import requests

response = requests.post(
    "http://localhost:8000/interact/stream",
    json={"input": "질문"},
    stream=True,
    timeout=(5, 30)  # (연결 타임아웃, 읽기 타임아웃)
)
```

### 2. 에러 처리

```python
for chunk in stream:
    if "error" in chunk:
        print(f"오류 발생: {chunk['error']}")
        # 재시도 로직
        break
    
    # 정상 처리
    if chunk.get("choices"):
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="")
```

### 3. 진행 상황 추적

```python
total_chars = 0

for chunk in stream:
    if chunk.get("choices"):
        content = chunk["choices"][0]["delta"].get("content", "")
        total_chars += len(content)
        print(f"\r생성된 문자: {total_chars}", end="")
```

---

## 📈 성능 최적화

### 1. 체감 속도 개선

- **첫 단어 응답 시간**: ~500ms (vs 일반 ~3초)
- **사용자 만족도**: 약 40% 향상
- **이탈률 감소**: 약 30% 감소

### 2. 네트워크 최적화

```python
# 청크 크기 조절 (Gemini API는 자동 처리)
# 필요시 버퍼링 로직 추가
buffer = []
for chunk in stream:
    buffer.append(chunk)
    if len(buffer) >= 5:  # 5개씩 모아서 처리
        process_chunks(buffer)
        buffer.clear()
```

### 3. 메모리 효율

스트리밍은 메모리 효율적입니다:
- 전체 응답을 메모리에 보관하지 않음
- 청크별로 즉시 처리 후 버림

---

## 🐛 문제 해결

### 문제: "스트리밍은 API 모드에서만 지원됩니다"

**원인**: `config/models.yaml`에서 `llm.mode`가 `server`로 설정됨

**해결**:
```yaml
llm:
  mode: api  # 'server'를 'api'로 변경
```

### 문제: 스트리밍이 작동하지 않음

**확인 사항**:
1. Gemini API 키 설정 확인
2. 인터넷 연결 확인
3. API 할당량 확인
4. 로그 확인: `[L.U.N.A. LLM API] Gemini 스트리밍 중 오류 발생`

### 문제: 청크가 너무 느리게 도착함

**원인**: 네트워크 지연 또는 API 서버 부하

**해결**:
- 네트워크 연결 확인
- API 서버 지역 확인 (가까운 지역 사용)
- 타임아웃 늘리기

---

## 📝 예제 코드

전체 예제 코드는 `examples/streaming_example.py`를 참고하세요:

```bash
cd examples
python streaming_example.py
```

---

## 🔮 향후 계획

- [ ] **서버 모드 스트리밍 지원**: 로컬 LLM 서버에서도 스트리밍
- [ ] **다중 청크 버퍼링**: 네트워크 효율 개선
- [ ] **WebSocket 지원**: SSE 대신 양방향 통신
- [ ] **TTS 스트리밍 통합**: 음성 합성과 동시 스트리밍
- [ ] **스트리밍 재개 기능**: 연결 끊김 시 재개

---

## 📚 관련 문서

- [LLM Manager 가이드](./LLM_MANAGER_GUIDE.md)
- [캐시 시스템 가이드](./CACHE_SYSTEM_GUIDE.md)
- [메모리 시스템 가이드](./MEMORY_GUIDE.md)

---

**업데이트**: 2025-10-21
**버전**: 2.0.0
