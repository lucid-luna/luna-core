# luna-core ↔ luna-plugin 연동 가이드

## 📋 개요

**luna-core**와 **luna-plugin**의 통합 구조입니다.

```
luna-plugin (플러그인 개발/관리)
├── sdk/
│   ├── server.py           → PluginMCPServer (플러그인 템플릿)
│   ├── manager.py          → PluginManager (플러그인 발견/로드/활성화)
│   ├── config.py           → 설정 로더
│   └── ...
├── plugins/
│   ├── echo/
│   │   ├── __init__.py
│   │   └── server.py       → MCP 서버 구현
│   └── ...
└── config/
    └── config.json         → 플러그인 런타임 설정

            ↑ (STDIO 통신)

luna-core (런타임)
├── services/mcp/
│   ├── external_manager.py → MCP 서버 프로세스 관리
│   ├── tool_manager.py     → 도구 수집 & ToolRegistry 연동
│   └── ...
├── main.py                 → 플러그인 로드 & HTTP API
└── config/
    └── mcp_servers.json    → 외부 MCP 서버 설정
```

---

## 🔗 연동 흐름

### 1. luna-core 시작

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. main.py의 AppLifespan.__aenter__()

```python
# 1️⃣ 플러그인 매니저 초기화
from sdk.manager import PluginManager
pm = PluginManager(str(plugins_path), plugin_config)
discovered = pm.discover_plugins()  # → ["echo", ...]
activated = pm.load_plugin("echo")  # → 로드

# 2️⃣ 외부 MCP 서버 시작 (config/mcp_servers.json 기반)
self.mcp_mgr = ExternalMCPManager(config_path="config/mcp_servers.json")
await self.mcp_mgr.start_enabled()
# → python -m plugins.echo.server 실행
# → STDIO 연결 수립

# 3️⃣ MCP 도구 자동 수집 & ToolRegistry 등록
self.tool_manager = MCPToolManager(self.mcp_mgr, tool_registry)
await self.tool_manager.initialize()
# → list_tools() 호출
# → ToolRegistry에 등록
```

### 3. HTTP API로 도구 호출

```bash
GET /mcp/tools
→ MCPToolManager.get_tool_list()
→ {"tools": [{"id": "echo/ping", ...}], "total": 1}

POST /mcp/call
→ MCPToolManager.call_tool("echo", "ping", {"text": "..."})
→ ExternalMCPManager가 STDIO를 통해 플러그인 서버에 전달
→ 플러그인이 결과 반환
```

---

## 🚀 플러그인 개발 & 배포

### 1. luna-plugin에서 플러그인 개발

#### 디렉토리 구조

```
luna-plugin/
└── plugins/
    └── myapp/
        ├── __init__.py
        └── server.py
```

#### server.py 구현

```python
from sdk.server import PluginMCPServer, run_server

mcp = PluginMCPServer("myapp", version="1.0.0")

@mcp.tool(rate="30/m")
def my_function(param: str) -> str:
    """도구 설명"""
    return f"Result: {param}"

if __name__ == "__main__":
    run_server(mcp)
```

#### 테스트 (로컬)

```bash
cd luna-plugin
python -m plugins.myapp.server
```

### 2. luna-core에 등록

#### config/mcp_servers.json 수정

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
            "id": "myapp",
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "plugins.myapp.server"],
            "enabled": true,
            "timeoutMs": 8000,
            "namespace": "myapp"
        }
    ]
}
```

#### luna-core 재시작

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 확인

```bash
curl http://localhost:8000/mcp/tools
# → myapp/my_function 도구가 포함됨
```

---

## 📂 파일 역할 정의

### luna-plugin 프로젝트

| 파일/폴더 | 책임 | 설명 |
|----------|------|------|
| `sdk/server.py` | 플러그인 개발자 | MCP 서버 템플릿 (FastMCP 래퍼) |
| `sdk/manager.py` | 플러그인 관리자 | 플러그인 발견/로드/활성화 |
| `sdk/config.py` | 설정 관리 | 플러그인 설정 로더 |
| `plugins/*/server.py` | 플러그인 개발자 | 각 플러그인의 MCP 서버 구현 |
| `config/config.json` | 플러그인 관리자 | 플러그인 런타임 설정 (discord token 등) |

### luna-core 프로젝트

| 파일/폴더 | 책임 | 설명 |
|----------|------|------|
| `services/mcp/external_manager.py` | core 개발자 | STDIO 기반 MCP 서버 생명주기 관리 |
| `services/mcp/tool_manager.py` | core 개발자 | 플러그인 도구 수집 & 라우팅 |
| `main.py` | core 개발자 | 플러그인 로드 & HTTP API |
| `config/mcp_servers.json` | core 관리자 | 외부 MCP 서버 시작 설정 |

---

## 🔄 플러그인 추가 절차

### Step 1: luna-plugin에서 개발

```bash
cd luna-plugin
mkdir -p plugins/myapp
touch plugins/myapp/__init__.py
cat > plugins/myapp/server.py << 'EOF'
from sdk.server import PluginMCPServer, run_server

mcp = PluginMCPServer("myapp", version="1.0.0")

@mcp.tool()
def my_tool(text: str) -> str:
    return f"Echo: {text}"

if __name__ == "__main__":
    run_server(mcp)
EOF
```

### Step 2: 로컬 테스트

```bash
python -m plugins.myapp.server
# 별도 터미널에서
curl -X POST http://localhost:5000/tools  # 만약 SSE 사용 시
```

### Step 3: luna-core의 mcp_servers.json에 추가

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

### Step 4: luna-core 재시작

```bash
cd luna-core
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 5: 도구 확인

```bash
curl http://localhost:8000/mcp/tools | jq '.tools[] | select(.name | contains("myapp"))'
```

---

## 🧪 테스트

### 통합 테스트

```bash
cd luna-core
python test_mcp_integration.py
```

**예상 출력:**
```
[4] 외부 MCP 서버 시작 (enabled=true인 서버만)...
✓ 외부 MCP 서버 시작 완료

[5] MCP 도구 동기화...
✓ MCP 도구 동기화 완료

[6] 등록된 MCP 도구 목록:

  1. echo/ping
     ID: echo/ping
     Description: Echo back the text you send.
```

### HTTP 테스트

```bash
python test_mcp_http.py
```

**예상 출력:**
```
[✓] 모든 테스트 통과!
```

---

## 📝 설정 파일 상세

### luna-plugin/config/config.json

```json
{
    "discord": {
        "token": "YOUR_TOKEN",
        "luna_api_url": "http://localhost:8000"
    },
    "backend": {
        "luna_core_api_url": "http://localhost:8000"
    },
    "plugins": ["discord"]
}
```

**로드 순서:**
1. 환경변수: `LUNA_PLUGIN_{plugin_key}_CONFIG` (JSON string)
2. 파일: `config/config.json`
3. 파일: `config/config_example.json`

### luna-core/config/mcp_servers.json

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
            "namespace": "echo",
            "cwd": null,
            "env": null
        }
    ]
}
```

**필드:**
- `id`: 서버 고유 ID
- `transport`: 통신 방식 (stdio만 지원)
- `command`: 실행 명령어
- `args`: 명령어 인자 배열
- `enabled`: 시작 시 자동 활성화 여부
- `timeoutMs`: 도구 호출 타임아웃
- `namespace`: 도구 네임스페이싱 (선택사항)
- `cwd`: 작업 디렉토리 (선택사항)
- `env`: 환경변수 (선택사항)

---

## 🐛 트러블슈팅

### 도구가 로드되지 않음

**증상:** `/mcp/tools` 응답이 빈 목록

**확인 사항:**
1. MCP 서버 프로세스 실행 확인
   ```bash
   ps aux | grep "plugins.echo.server"
   ```

2. 로그에서 "[MCP]" 필터링
   ```bash
   # 서버 로그 보기
   uvicorn main:app ... 2>&1 | grep "\[MCP\]"
   ```

3. config/mcp_servers.json 확인
   ```bash
   cat config/mcp_servers.json | jq '.servers[] | .enabled'
   ```

4. 플러그인 서버가 STDIO 응답 가능한지 테스트
   ```bash
   cd luna-plugin
   python -m plugins.echo.server
   # 별도 터미널에서 STDIN에 MCP 메시지 전송
   ```

### "MCP Tool Manager not initialized"

**원인:** ExternalMCPManager나 MCPToolManager가 초기화되지 않음

**해결:**
1. `sdk.manager` import 확인
   ```bash
   cd luna-plugin
   python -c "from sdk.manager import PluginManager; print('OK')"
   ```

2. main.py 로그 확인
   ```
   [플러그인] 매니저 임포트 실패(sdk.manager): ...
   ```

3. sys.path 확인
   ```python
   import sys
   print([p for p in sys.path if 'luna-plugin' in p])
   ```

### 플러그인 명령어 실행 실패

**원인:** `command`와 `args`가 잘못되었을 수 있음

**올바른 예:**
```json
{
    "command": "python",
    "args": ["-m", "plugins.echo.server"]
}
```

**주의:** 
- 절대 경로가 아닌 모듈 이름 사용
- `cwd` 필드로 작업 디렉토리 명시 가능

---

## 📚 관련 파일

- `luna-core/docs/MCP_INTEGRATION_GUIDE.md` — MCP 통합 전체 가이드
- `luna-core/docs/MCP_PLUGIN_IMPLEMENTATION_REPORT.md` — 구현 보고서
- `luna-core/test_mcp_integration.py` — 비동기 통합 테스트
- `luna-core/test_mcp_http.py` — HTTP API 테스트
- `luna-plugin/sdk/manager.py` — PluginManager 구현
- `luna-plugin/sdk/server.py` — PluginMCPServer 템플릿

---

## 🎯 다음 단계

- [ ] LLM 통합 (Phase 2)
- [ ] 도구 체인 실행
- [ ] 도구별 권한 관리
- [ ] 플러그인 마켓플레이스 (선택)
