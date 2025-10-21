# L.U.N.A. LLM Manager 사용 가이드

L.U.N.A. 프로젝트에서 로컬 LLM 서버와 Gemini API를 선택적으로 사용할 수 있는 통합 인터페이스입니다.

## 주요 기능

- **두 가지 모드 지원**
  - `server`: 로컬 LLM 서버 (llama.cpp 등)
  - `api`: Gemini API 등의 클라우드 LLM API

- **통일된 인터페이스**: 모드에 관계없이 동일한 방식으로 사용
- **자동/대화형 선택**: 설정 파일 또는 사용자 선택으로 모드 결정

## 설치

필요한 패키지 설치:

```bash
# 로컬 서버 모드
pip install requests pyyaml

# Gemini API 모드
pip install google-genai pyyaml
```

## 환경 변수 설정

### API 모드 (Gemini)
```bash
# Windows (PowerShell)
$env:LLM_MODE="api"
$env:GEMINI_API_KEY="your-api-key-here"

# Linux/Mac
export LLM_MODE="api"
export GEMINI_API_KEY="your-api-key-here"
```

### 서버 모드 (로컬 LLM)
```bash
# Windows (PowerShell)
$env:LLM_MODE="server"
$env:LLM_SERVER_URL="http://localhost:8080"

# Linux/Mac
export LLM_MODE="server"
export LLM_SERVER_URL="http://localhost:8080"
```

## 설정 파일

`config/models.yaml` 파일에서 설정:

```yaml
llm:
  # 기본 모드 설정
  mode: "server"  # 또는 "api"
  
  # 로컬 서버 설정
  servers:
    luna:
      url: "http://localhost:8080"
      alias: "luna-model"
  
  # API 설정
  api:
    gemini:
      model: "gemini-2.0-flash-exp"
```

## 사용 방법

### 방법 1: 빠른 시작 (자동 모드)

```python
from utils.llm_config import get_llm_manager

# 환경 변수 또는 설정 파일 기반으로 자동 초기화
llm, target = get_llm_manager(auto_mode=True)

response = llm.generate(
    target=target,
    system_prompt="You are Luna, a helpful AI assistant.",
    user_prompt="안녕하세요!"
)

# 응답 추출
content = response["choices"][0]["message"]["content"]
print(f"Luna: {content}")
```

### 방법 2: 대화형 모드 선택

```python
from utils.llm_config import get_llm_manager

# 사용자가 직접 모드 선택
llm, target = get_llm_manager(auto_mode=False)

response = llm.generate(
    target=target,
    system_prompt="You are Luna.",
    user_prompt="Hello!"
)
```

### 방법 3: 직접 초기화

#### 로컬 서버 모드
```python
from services.llm_manager import LLMManager

server_configs = {
    "luna": {
        "url": "http://localhost:8080",
        "alias": "luna-model"
    }
}

llm = LLMManager(mode="server", server_configs=server_configs)

response = llm.generate(
    target="luna",
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello!",
    temperature=0.9,
    max_tokens=128
)
```

#### Gemini API 모드
```python
from services.llm_manager import LLMManager

api_configs = {
    "gemini": {
        "api_key": "your-api-key",
        "model": "gemini-2.0-flash-exp"
    }
}

llm = LLMManager(mode="api", api_configs=api_configs)

response = llm.generate(
    target="gemini",
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello!",
    model="gemini-2.0-flash-exp"  # 선택적
)
```

### 방법 4: 대화 내역 사용

```python
messages = [
    {"role": "system", "content": "You are Luna."},
    {"role": "user", "content": "안녕하세요!"},
    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
    {"role": "user", "content": "날씨가 어때요?"}
]

response = llm.generate(
    target=target,
    messages=messages
)
```

## 응답 형식

두 모드 모두 동일한 OpenAI 스타일 응답 형식:

```python
{
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "응답 텍스트"
            },
            "finish_reason": "stop"
        }
    ],
    "model": "모델명",
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
}
```

## 예제 실행

```bash
# 빠른 시작 예제
python examples/quick_start.py

# 상세 예제
python examples/llm_manager_example.py
```

## 모드별 차이점

| 기능 | 서버 모드 | API 모드 |
|------|----------|---------|
| temperature | ✅ 지원 | ❌ 미지원 |
| max_tokens | ✅ 지원 | ❌ 미지원 |
| tools | ✅ 지원 | ❌ 미지원 |
| 비용 | 무료 (로컬) | 유료 (API) |
| 속도 | 하드웨어 의존 | 빠름 |
| 오프라인 | ✅ 가능 | ❌ 불가능 |

## 주의사항

1. **API 키 보안**: API 키는 환경 변수로 관리하세요. 코드나 설정 파일에 직접 넣지 마세요.
2. **모드 전환**: 런타임 중 모드 전환은 불가능합니다. 새로운 인스턴스를 생성하세요.
3. **에러 처리**: 응답에 `"error"` 키가 있는지 확인하세요.

## 트러블슈팅

### Gemini API 오류
```python
# API 키 확인
import os
print(os.getenv("GEMINI_API_KEY"))
```

### 로컬 서버 연결 오류
```bash
# 서버 상태 확인
curl http://localhost:8080/health
```

## 라이센스

L.U.N.A. 프로젝트 라이센스를 따릅니다.
