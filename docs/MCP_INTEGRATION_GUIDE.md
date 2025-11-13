# L.U.N.A. MCP (Model Context Protocol) 플러그인 통합 가이드

## 📋 개요

이 가이드는 **luna-core에서 MCP 플러그인을 로드하고 도구를 호출하는 방법**을 설명합니다.

> ⚠️ **주의**: 이 구현은 **LLM 통합 이전 단계**입니다. 현재는 HTTP API로만 도구를 호출할 수 있습니다.

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      luna-core                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ AppLifespan.__aenter__()                             │   │
│  │                                                      │   │
│  │ 1. ExternalMCPManager 생성                          │   │
│  │    └─ config/mcp_servers.json 로드                  │   │
│  │                                                      │   │
│  │ 2. MCP 서버 시작 (STDIO)                            │   │
│  │    └─ "python -m plugins.echo.server" 등 실행       │   │
│  │                                                      │   │
│  │ 3. MCPToolManager 생성                             │   │
│  │    └─ list_tools() 호출해서 ToolRegistry에 등록     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ HTTP 엔드포인트 (FastAPI)                           │   │
│  │ GET  /mcp/tools          → 도구 목록 조회           │   │
│  │ POST /mcp/call           → 도구 호출                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ToolRegistry (InteractionService용)                 │   │
│  │ └─ 로컬 + 외부 도구 모두 저장                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 설정

### 1. MCP 서버 설정: `config/mcp_servers.json`

```json
{
    "servers": [
        {
            "id": "echo",
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "plugins.echo.server"],
            "enabled": true,
            "timeoutMs": 8000,
            "namespace": "echo"
        },
        {
            "id": "spotify",
            "transport": "stdio",
            "command": "python",
            "args": ["../luna-plugin/plugins/spotify/server.py"],
            "enabled": false,
            "timeoutMs": 8000,
            "namespace": "spotify"
        }
    ]
}
```

**필드 설명:**

| 필드 | 설명 | 예시 |
|------|------|------|
| `id` | 고유 서버 ID | `"echo"`, `"spotify"` |
| `transport` | 통신 방식 (현재 STDIO만 지원) | `"stdio"` |
| `command` | 실행할 명령어 | `"python"`, `"node"` |
| `args` | 명령어 인자 배열 | `["-m", "plugins.echo.server"]` |
| `enabled` | 시작 시 활성화 여부 | `true`, `false` |
| `timeoutMs` | 도구 호출 타임아웃 (ms) | `8000` |
| `namespace` | 도구 네임스페이싱 (선택사항) | `"echo"` |

---

## 🚀 사용 방법

### 1. 서버 시작

```bash
# luna-core 디렉토리에서
uvicorn main:app --host 0.0.0.0 --port 8000
```

**출력:**
```
[MCP] 외부 MCP 서버(ENABLED=true) 시작 완료
[MCP] MCP 도구 매니저 초기화 완료
```

### 2. HTTP API로 도구 조회

```bash
curl http://localhost:8000/mcp/tools
```

**응답:**
```json
{
  "tools": [
    {
      "id": "echo/ping",
      "name": "echo/ping",
      "description": "Echo back the text you send.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "text": {
            "type": "string"
          }
        }
      }
    }
  ],
  "total": 1
}
```

### 3. HTTP API로 도구 호출

```bash
curl -X POST http://localhost:8000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "echo",
    "tool_name": "ping",
    "arguments": {
      "text": "Hello, MCP!"
    }
  }'
```

**응답:**
```json
{
  "success": true,
  "result": "Hello, MCP!",
  "error": null
}
```

---

## 🧪 테스트

### 1. 비동기 통합 테스트

```bash
cd luna-core
python test_mcp_integration.py
```

**출력:**
```
================================================================================
L.U.N.A. MCP 플러그인 로드 테스트
================================================================================

[1] ExternalMCPManager 초기화...
[2] ToolRegistry 초기화...
[3] MCPToolManager 초기화...

[4] 외부 MCP 서버 시작 (enabled=true인 서버만)...
✓ 외부 MCP 서버 시작 완료

[5] MCP 도구 동기화...
✓ MCP 도구 동기화 완료

[6] 등록된 MCP 도구 목록:

  1. echo/ping
     ID: echo/ping
     Description: Echo back the text you send.

[7] 도구 호출 테스트:

  테스트 도구: echo/ping
  Server ID: echo
  Tool Name: ping
  Arguments: {'text': 'Hello from MCP Test!'}
  ✓ 호출 성공!
  Result: Hello from MCP Test!

[8] ToolRegistry 내용:
  - echo/ping

[9] MCP 서버 종료...
✓ 종료 완료

================================================================================
테스트 완료!
================================================================================
```

### 2. HTTP 엔드포인트 테스트

```bash
cd luna-core
python test_mcp_http.py
```

**출력:**
```
================================================================================
L.U.N.A. MCP HTTP 엔드포인트 테스트
================================================================================

Base URL: http://localhost:8000

[✓] 서버 연결 성공

[*] 헬스 체크
Status: 200
Response: {
  "server": "L.U.N.A.",
  "version": "1.3.0",
  "status": "healthy"
}

[*] MCP 도구 목록 조회
Status: 200

총 도구 개수: 1

등록된 도구:

  1. echo/ping
     ID: echo/ping
     Description: Echo back the text you send.

...

================================================================================
테스트 결과 요약
================================================================================
✓ PASS Health Check
✓ PASS Get MCP Tools
✓ PASS Call Tool (echo/ping)
✓ PASS Call Invalid Tool

총합: 4 성공, 0 실패

[✓] 모든 테스트 통과!
```

---

## 📚 클래스 문서

### `MCPToolManager`

**위치:** `services/mcp/tool_manager.py`

ExternalMCPManager와 ToolRegistry를 통합하여 외부 MCP 플러그인의 도구들을 관리합니다.

#### 주요 메서드

```python
async def initialize()
    """
    ExternalMCPManager의 모든 활성화된 서버에서
    도구 목록을 수집하고 ToolRegistry에 등록합니다.
    """

async def call_tool(server_id: str, tool_name: str, arguments: dict)
    """특정 서버의 도구를 호출합니다."""

def get_tool_list() -> List[Dict[str, Any]]
    """모든 등록된 도구의 목록을 반환합니다."""

def get_tool_info(server_id: str, tool_name: str) -> Optional[types.Tool]
    """특정 도구의 정보를 조회합니다."""

async def reload()
    """도구 목록을 다시 수집합니다."""
```

### `ExternalMCPManager`

**위치:** `services/mcp/external_manager.py`

STDIO 기반 외부 MCP 서버들의 생명 주기를 관리합니다.

#### 주요 메서드

```python
def load_config()
    """config/mcp_servers.json에서 설정을 로드합니다."""

async def start_enabled()
    """enabled=true인 모든 서버를 시작합니다."""

async def start(server_id: str)
    """특정 서버를 시작합니다."""

async def stop(server_id: str)
    """특정 서버를 중지합니다."""

async def list_tools(server_id: str) -> List[types.Tool]
    """서버의 도구 목록을 조회합니다."""

async def call_tool(server_id: str, name: str, arguments: dict)
    """서버의 도구를 호출합니다."""
```

### `ToolRegistry`

**위치:** `services/mcp/tool_registry.py`

로컬 + 외부 도구들을 등록하고 호출합니다.

#### 주요 메서드

```python
def register(name: str, func: Callable[..., Any])
    """도구를 등록합니다."""

def unregister(name: str)
    """도구를 제거합니다."""

def list() -> list[str]
    """모든 등록된 도구 이름을 반환합니다."""

def has(name: str) -> bool
    """도구가 존재하는지 확인합니다."""

def call(name: str, **kwargs) -> Any
    """도구를 호출합니다."""
```

---

## 🔗 HTTP 엔드포인트

### `GET /mcp/tools`

모든 등록된 MCP 도구 목록을 반환합니다.

**Response:**
```json
{
  "tools": [
    {
      "id": "echo/ping",
      "name": "echo/ping",
      "description": "Echo back the text you send.",
      "inputSchema": {...}
    }
  ],
  "total": 1
}
```

### `POST /mcp/call`

MCP 도구를 호출합니다.

**Request:**
```json
{
  "server_id": "echo",
  "tool_name": "ping",
  "arguments": {
    "text": "Hello!"
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": "Hello!",
  "error": null
}
```

---

## 🔧 플러그인 개발

새로운 MCP 플러그인을 만드는 방법:

### 1. 플러그인 구조

```
luna-plugin/plugins/myapp/
├── __init__.py
└── server.py
```

### 2. 플러그인 구현 (server.py)

```python
from sdk.server import PluginMCPServer, run_server

mcp = PluginMCPServer("myapp", version="1.0.0")

@mcp.tool(rate="30/m")
def my_tool(param: str) -> str:
    """도구 설명"""
    return f"Result: {param}"

if __name__ == "__main__":
    run_server(mcp)
```

### 3. 설정 추가 (config/mcp_servers.json)

```json
{
    "id": "myapp",
    "transport": "stdio",
    "command": "python",
    "args": ["-m", "plugins.myapp.server"],
    "enabled": true,
    "timeoutMs": 8000,
    "namespace": "myapp"
}
```

---

## 📖 트러블슈팅

### 도구가 로드되지 않음

1. **설정 확인:**
   ```bash
   cat config/mcp_servers.json
   ```

2. **서버 로그 확인:**
   ```
   [MCP] echo: 1개 도구 발견
   ```

3. **enabled 확인:**
   ```json
   "enabled": true
   ```

### 도구 호출이 실패함

1. **arguments 확인:**
   ```bash
   GET /mcp/tools  # inputSchema 확인
   ```

2. **타임아웃 확인:**
   ```json
   "timeoutMs": 8000  # 필요하면 증가
   ```

3. **서버 로그:**
   ```
   [MCPToolManager] 도구 호출 실패: ...
   ```

---

## 🎯 다음 단계

### Phase 2: LLM 통합
- [ ] LLMManager에 MCPToolManager 주입
- [ ] LLM 프롬프트에 도구 정보 포함
- [ ] LLM 출력에서 도구 호출 파싱

### Phase 3: 고급 기능
- [ ] 도구 호출 캐싱
- [ ] 도구 체인 (chain of thought)
- [ ] 도구별 인증/권한 관리

---

## 📞 참고

- **MCP 공식 문서:** https://modelcontextprotocol.io/
- **FastMCP 문서:** https://github.com/jlowin/fastmcp
- **Luna Plugin SDK:** `luna-plugin/sdk/server.py`
