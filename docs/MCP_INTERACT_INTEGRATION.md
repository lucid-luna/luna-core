# MCP 도구 통합 가이드 - `/interact` 엔드포인트

## 개요

`/interact` 엔드포인트는 이제 **선택적 MCP 도구 사용**을 지원합니다. 사용자는 요청에 `use_tools` 플래그를 추가하여 LLM이 자동으로 MCP 도구를 호출하도록 할 수 있습니다.

## API 사용법

### 기본 요청 (도구 없음)

```bash
curl -X POST http://localhost:8000/interact \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요",
    "use_tools": false
  }'
```

**응답:**
```json
{
  "text": "음, 다엘. 또 나 부르는 거야?",
  "emotion": "neutral",
  "intent": "general",
  "style": "Neutral",
  "audio_url": "/outputs/xxx.wav"
}
```

### MCP 도구를 사용한 요청

```bash
curl -X POST http://localhost:8000/interact \
  -H "Content-Type: application/json" \
  -d '{
    "input": "15와 20을 더해줘",
    "use_tools": true
  }'
```

**응답:**
```json
{
  "text": "계산해볼까. 그럼 35네.",
  "emotion": "neutral",
  "intent": "agent",
  "style": "Neutral",
  "audio_url": "/outputs/xxx.wav"
}
```

## 요청 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|-------|------|
| `input` | string | ✅ | - | 사용자 입력 (한국어/영어) |
| `use_tools` | boolean | ❌ | `false` | MCP 도구 사용 여부 |

## 응답 포맷

```json
{
  "text": "응답 텍스트",
  "emotion": "감정 (neutral, joy, anger, ...)",
  "intent": "의도 (general, agent, ...)",
  "style": "TTS 스타일 (Neutral, Happy, ...)",
  "audio_url": "생성된 음성 파일 경로"
}
```

## 사용 가능한 MCP 도구

현재 등록된 도구들:

- **`simple/add`** - 두 정수 더하기
  ```
  입력: {"a": 10, "b": 20}
  출력: 30
  ```

- **`echo/ping`** - 텍스트 에코
  ```
  입력: {"text": "hello"}
  출력: "hello"
  ```

- 기타 등록된 MCP 도구들 (`/mcp/tools` 엔드포인트로 확인)

## 동작 흐름

### 일반 모드 (`use_tools: false`)
```
사용자 입력
    ↓
LLM (도구 없음)
    ↓
응답
```

### 에이전트 모드 (`use_tools: true`)
```
사용자 입력
    ↓
LLM (도구 목록 제공)
    ↓
도구 호출 필요?
    ├─ YES → MCP 도구 실행 → LLM에 결과 전달 → 응답
    └─ NO → 응답
```

## 예제

### 예제 1: 계산

**요청:**
```bash
curl -X POST http://localhost:8000/interact \
  -H "Content-Type: application/json" \
  -d '{
    "input": "50 더하기 30 계산해줘",
    "use_tools": true
  }'
```

**응답:**
```json
{
  "text": "계산해볼까. 그럼 80이네.",
  "emotion": "neutral",
  "intent": "agent",
  "style": "Neutral",
  "audio_url": "/outputs/xxx.wav"
}
```

### 예제 2: 일반 대화

**요청:**
```bash
curl -X POST http://localhost:8000/interact \
  -H "Content-Type: application/json" \
  -d '{
    "input": "오늘 날씨 어때?",
    "use_tools": false
  }'
```

**응답:**
```json
{
  "text": "글쎄 모르겠는데, 밖을 봐봐.",
  "emotion": "neutral",
  "intent": "general",
  "style": "Neutral",
  "audio_url": "/outputs/xxx.wav"
}
```

## PowerShell 테스트

```powershell
# UTF-8 인코딩 설정
chcp 65001

# 일반 모드
$body = @{input="안녕하세요"; use_tools=$false} | ConvertTo-Json
curl -X POST http://localhost:8000/interact `
  -H "Content-Type: application/json" `
  -d $body

# 에이전트 모드
$body = @{input="15와 20을 더해줘"; use_tools=$true} | ConvertTo-Json
curl -X POST http://localhost:8000/interact `
  -H "Content-Type: application/json" `
  -d $body
```

## 주의사항

1. **`use_tools: true`일 때 MCP 서버가 없으면** - 일반 모드로 자동 전환
2. **도구 호출 실패 시** - LLM이 에러 메시지를 포함하여 응답
3. **응답 시간** - 도구 호출로 인해 일반 모드보다 약간 더 걸릴 수 있음
4. **한글 입력** - PowerShell은 `chcp 65001`로 UTF-8 설정 필요

## 관련 엔드포인트

- `GET /mcp/tools` - 사용 가능한 도구 목록 조회
- `POST /mcp/call` - 특정 도구 직접 호출
- `GET /mcp/external/config` - 등록된 MCP 서버 설정 조회
