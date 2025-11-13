# L.U.N.A. MCP 플러그인 로드 구현 완료 보고서

**작성일:** 2025-11-04  
**상태:** ✅ 완료  
**범위:** luna-core에서 MCP 플러그인을 불러오고 도구를 호출할 수 있는 기능

---

## 📋 구현 내용

### 1️⃣ **MCPToolManager 클래스** (`services/mcp/tool_manager.py`)

ExternalMCPManager와 ToolRegistry를 통합하는 통합 관리자입니다.

**주요 기능:**
- ✅ MCP 서버 시작 후 도구 목록 자동 수집
- ✅ ToolRegistry에 도구 자동 등록
- ✅ 도구 네임스페이싱 (예: `echo/ping`)
- ✅ 비동기 도구 호출 및 라우팅
- ✅ 도구 정보 조회

**클래스 구조:**
```python
class MCPToolManager:
    async def initialize()              # 도구 동기화
    async def reload()                  # 도구 목록 재로드
    async def call_tool()               # 도구 호출
    def get_tool_list()                 # 도구 목록 조회
    def get_tool_info()                 # 도구 정보 조회
    async def list_resources()          # 리소스 조회
```

---

### 2️⃣ **main.py 수정**

#### a) MCPToolManager 임포트 및 초기화
```python
from services.mcp.tool_manager import MCPToolManager

# AppLifespan.__aenter__()에서
self.tool_manager = MCPToolManager(self.mcp_mgr, tool_registry, logger=logger)
await self.tool_manager.initialize()
```

#### b) HTTP 엔드포인트 추가

**GET `/mcp/tools`** - 도구 목록 조회
- 모든 등록된 MCP 도구 반환
- 네임스페이싱된 이름 포함

**POST `/mcp/call`** - 도구 호출
- 특정 서버의 도구 호출
- 인자 및 결과 처리

---

### 3️⃣ **Request/Response 스키마**

```python
class MCPToolCallRequest(BaseModel):
    server_id: str      # "echo"
    tool_name: str      # "ping"
    arguments: dict     # {"text": "..."}

class MCPToolCallResponse(BaseModel):
    success: bool
    result: dict | Any
    error: str | None

class MCPToolInfo(BaseModel):
    id: str             # "echo/ping"
    name: str           # "echo/ping"
    description: str
    inputSchema: dict

class MCPToolListResponse(BaseModel):
    tools: list[MCPToolInfo]
    total: int
```

---

### 4️⃣ **테스트 스크립트**

#### `test_mcp_integration.py` - 비동기 통합 테스트
```bash
python test_mcp_integration.py
```

**테스트 항목:**
- [x] ExternalMCPManager 초기화
- [x] MCP 서버 시작
- [x] MCPToolManager 도구 동기화
- [x] 도구 목록 조회
- [x] 도구 호출
- [x] ToolRegistry 확인

#### `test_mcp_http.py` - HTTP 엔드포인트 테스트
```bash
python test_mcp_http.py
```

**테스트 항목:**
- [x] Health Check
- [x] 도구 목록 조회 (`GET /mcp/tools`)
- [x] 도구 호출 (`POST /mcp/call`)
- [x] 잘못된 도구 호출 에러 처리

---

### 5️⃣ **설정 파일 업데이트**

#### `config/mcp_servers.json`
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
        }
    ]
}
```

**필드 설명:**
| 필드 | 설명 |
|------|------|
| `id` | 서버 고유 ID |
| `transport` | 통신 방식 (STDIO만 지원) |
| `command` | 실행 명령어 |
| `args` | 명령어 인자 배열 |
| `enabled` | 시작 시 활성화 여부 |
| `timeoutMs` | 도구 호출 타임아웃 |
| `namespace` | 도구 네임스페이싱 (선택사항) |

---

### 6️⃣ **문서**

#### `docs/MCP_INTEGRATION_GUIDE.md`
- 📖 전체 아키텍처 설명
- 📝 설정 가이드
- 🚀 사용 방법
- 🧪 테스트 실행
- 📚 클래스 문서
- 🔗 HTTP 엔드포인트
- 🔧 플러그인 개발 가이드
- 📖 트러블슈팅

---

## 🔄 동작 흐름

### 시작 시
```
1. luna-core 시작
   ↓
2. AppLifespan.__aenter__() 호출
   ├─ ExternalMCPManager 생성
   ├─ config/mcp_servers.json 로드
   ├─ enabled=true인 MCP 서버 시작 (STDIO)
   │  └─ python -m plugins.echo.server
   ├─ MCPToolManager 생성
   ├─ ExternalMCPManager에서 list_tools() 호출
   ├─ 각 도구마다 ToolRegistry에 등록
   └─ 모든 초기화 완료
   ↓
3. HTTP 엔드포인트 활성화
   ├─ GET /mcp/tools
   └─ POST /mcp/call
```

### 도구 호출 시
```
1. POST /mcp/call
   {
     "server_id": "echo",
     "tool_name": "ping",
     "arguments": {"text": "Hello"}
   }
   ↓
2. main.py 핸들러
   ├─ MCPToolManager.call_tool() 호출
   └─ await tool_manager.call_tool(
        server_id="echo",
        tool_name="ping",
        arguments={"text": "Hello"}
      )
   ↓
3. MCPToolManager
   ├─ ExternalMCPManager.call_tool() 호출
   └─ 결과 반환
   ↓
4. HTTP 응답
   {
     "success": true,
     "result": "Hello",
     "error": null
   }
```

---

## 📊 파일 구조

```
luna-core/
├── services/mcp/
│   ├── tool_manager.py          ✅ NEW
│   ├── external_manager.py      (기존, 수정 없음)
│   ├── tool_registry.py         (기존, 수정 없음)
│   ├── types.py                 (기존)
│   └── internal_server.py       (기존)
│
├── main.py                      ✅ MODIFIED
│   ├── MCPToolManager 임포트
│   ├─ MCPToolManager 초기화
│   └─ HTTP 엔드포인트 추가
│
├── config/
│   └── mcp_servers.json         ✅ MODIFIED
│       └─ namespace 필드 추가
│
├── docs/
│   └── MCP_INTEGRATION_GUIDE.md  ✅ NEW
│
├── test_mcp_integration.py      ✅ NEW
└── test_mcp_http.py             ✅ NEW
```

---

## ✅ 완료된 작업

- [x] MCPToolManager 클래스 구현
  - [x] 도구 동기화
  - [x] 도구 호출 라우팅
  - [x] 도구 목록 조회
  - [x] 네임스페이싱

- [x] main.py 수정
  - [x] MCPToolManager 임포트 및 초기화
  - [x] HTTP 엔드포인트 추가
  - [x] Request/Response 스키마 정의

- [x] 테스트 스크립트
  - [x] 통합 테스트 (test_mcp_integration.py)
  - [x] HTTP 테스트 (test_mcp_http.py)

- [x] 설정 파일 업데이트
  - [x] namespace 필드 추가

- [x] 문서
  - [x] 통합 가이드 작성

---

## 🚀 사용 예시

### 1. 서버 시작
```bash
cd luna-core
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. 도구 목록 확인
```bash
curl http://localhost:8000/mcp/tools | jq
```

### 3. 도구 호출
```bash
curl -X POST http://localhost:8000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "echo",
    "tool_name": "ping",
    "arguments": {"text": "Hello, MCP!"}
  }' | jq
```

### 4. 테스트 실행
```bash
# 통합 테스트
python test_mcp_integration.py

# HTTP 테스트
python test_mcp_http.py
```

---

## 📝 주요 특징

✅ **자동 도구 발견**
- MCP 서버 시작 후 도구를 자동으로 수집

✅ **네임스페이싱**
- 여러 서버의 도구를 명확히 구분
- 예: `echo/ping`, `spotify/play`

✅ **비동기 처리**
- async/await으로 논블로킹 동작

✅ **에러 처리**
- 도구 호출 실패 시 명확한 에러 메시지

✅ **HTTP API**
- REST API로 도구 조회 및 호출

✅ **확장성**
- 새로운 MCP 서버 추가 시 설정만 변경하면 자동 로드

---

## 🎯 다음 단계 (Phase 2)

### LLM 통합
- [ ] LLMManager에 MCPToolManager 주입
- [ ] LLM 프롬프트에 도구 정보 포함
- [ ] LLM 출력에서 도구 호출 파싱 및 실행

### 고급 기능
- [ ] 도구 호출 캐싱
- [ ] 도구 체인 실행
- [ ] 도구별 권한 관리

---

## 🔍 확인 사항

- ✅ ExternalMCPManager가 올바르게 작동
- ✅ ToolRegistry에 도구가 등록됨
- ✅ HTTP 엔드포인트가 응답함
- ✅ 도구 호출이 정상 작동
- ✅ 에러 처리가 기능함
- ✅ 문서가 완전함

---

## 📞 기술 지원

**문제 발생 시:**
1. 로그 확인: `[MCP]` 태그로 필터링
2. 설정 확인: `config/mcp_servers.json`
3. 테스트 실행: `python test_mcp_integration.py`
4. 가이드 참고: `docs/MCP_INTEGRATION_GUIDE.md`

---

**상태:** ✅ 준비 완료  
**테스트:** ✅ 통과  
**문서:** ✅ 작성 완료
