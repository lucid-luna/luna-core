# ====================================================================
#  File: main.py
# ====================================================================
"""
L.U.N.A. Core Backend Entry Point

• /ws/asr (WebSocket): 실시간 음성 스트림을 입력받아 전체 상호작용 파이프라인을 실행하고, 최종 결과를 반환합니다.
• /interact (POST)    : 텍스트 입력을 통해 전체 상호작용 파이프라인을 테스트합니다.
• /health (GET)       : 서비스의 현재 상태를 확인합니다.
• /synthesize (POST)  : 텍스트를 음성으로 합성합니다. (TTS)
• /translate (POST)   : 텍스트를 번역합니다.
• /analyze/vision (POST): 이미지를 분석하고 답변을 생성합니다.

실행 예시:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from typing import Any, Optional, List

import logging
import json
import yaml
import urllib.request
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Body
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.params import File
from pydantic import BaseModel
import torch

from services.asr import ASRService
from services.emotion import EmotionService
from services.multi_intent import MultiIntentService
from services.vision import VisionService
from services.tts import TTSService
from services.translator import TranslatorService
from services.llm_manager import LLMManager
from services.mcp.tool_registry import ToolRegistry
from services.memory import MemoryService
from services.interaction import InteractionService, InteractResponse
from utils.llm_config import create_llm_manager

from services.mcp.internal_server import LunaMCPInternal
from services.mcp.external_manager import ExternalMCPManager
from services.mcp.tool_manager import MCPToolManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LUNA_API")

# 서비스 인스턴스 생성
asr_service = None
emotion_service = None
multi_intent_service = None
vision_service = None
tts_service = None
translator_service = None
llm_service = None
memory_service = None
interaction_service = None

class AppLifespan:
    def __init__(self):
        # 필요하면 여기에 공유 상태를 넣을 수 있어요
        pass

    async def __aenter__(self):
        # ==== 여기에 기존 lifespan의 초기화 코드 전부 복사 ====
        # (global asr_service, ... 선언 포함)
        global asr_service, emotion_service, multi_intent_service, vision_service, tts_service, translator_service, llm_service, memory_service, interaction_service

        print("[L.U.N.A. Startup] 서비스 초기화 시작...")

        # --- 이하 기존 초기화 코드 그대로 ---
        asr_service = ASRService()
        emotion_service = EmotionService()
        multi_intent_service = MultiIntentService()
        vision_service = VisionService()
        translator_service = TranslatorService()

        print("[L.U.N.A. Startup] LLM 서비스 초기화 중...")
        llm, llm_target = create_llm_manager(config_path="config/models.yaml")
        if llm is None:
            print("[L.U.N.A. Startup] 경고: LLM 서비스 초기화 실패. 기본 서버 모드로 대체합니다.")
            from services.llm import LLMService
            llm = LLMService(server_configs={
                "rp": {"url": "http://localhost:8080", "alias": "Luna"},
                "translator": {"url": "http://localhost:8081", "alias": "Translator"}
            })
            llm_target = "rp"
        self.llm_service = llm
        llm_service = self.llm_service

        print(f"[L.U.N.A. Startup] LLM 모드: {self.llm_service.get_mode()}, 타겟: {llm_target}")

        print("[L.U.N.A. Startup] 메모리 서비스 초기화 중...")
        from pathlib import Path
        import yaml
        config_path = Path("config/models.yaml")
        memory_config = {
            "max_entries": 50,
            "max_context_turns": 6,
            "max_context_tokens": 1500,
            "summary_threshold": 20,
            "enable_auto_summary": True
        }
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config and "memory" in config:
                        memory_config.update(config["memory"])
                        print(f"[L.U.N.A. Startup] 메모리 설정 로드 완료")
            except Exception as e:
                print(f"[L.U.N.A. Startup] 메모리 설정 로드 실패 (기본값 사용): {e}")

        memory_service = MemoryService(
            memory_dir="./memory",
            max_entries=memory_config["max_entries"],
            max_context_turns=memory_config["max_context_turns"],
            max_context_tokens=memory_config.get("max_context_tokens", 1500),
            summary_threshold=memory_config["summary_threshold"],
            enable_auto_summary=memory_config["enable_auto_summary"],
            llm_service=llm_service
        )
        try:
            stats = memory_service.get_memory_stats()
            print(f"[L.U.N.A. Startup] 메모리 서비스 초기화 완료 (저장된 대화: {stats['conversations']}개, 요약: {stats['summaries']}개)")
        except Exception as e:
            print(f"[L.U.N.A. Startup] 메모리 서비스 초기화 완료 (통계 조회 실패: {e})")

        print("[L.U.N.A. Startup] TTS 서비스 초기화 중...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_service = TTSService(device=device)

        print("[L.U.N.A. Startup] TTS 웜업 시작...")
        try:
            _ = tts_service.synthesize(text="テスト", style="Neutral", style_weight=1.0)
            print("[L.U.N.A. Startup] TTS 웜업 완료!")
        except Exception as e:
            print(f"[L.U.N.A. Startup] TTS 웜업 실패: {e}")

        tool_registry = ToolRegistry()
        interaction_service = InteractionService(
            emotion_service=emotion_service,
            multi_intent_service=multi_intent_service,
            translator_service=translator_service,
            llm_service=llm_service,
            tts_service=tts_service,
            tool_registry=tool_registry,
            memory_service=memory_service,
            mcp_tool_manager=None,  # 나중에 설정됨
            llm_target=llm_target,
            logger=logger
        )
        self.interaction_service = interaction_service

        logger.info("[L.U.N.A. Startup] 모든 서비스 초기화 완료!")
        print("[L.U.N.A. Startup] 모든 서비스 초기화 완료!")

        # ===== 플러그인 매니저 초기화 (기존 코드 그대로) =====
        import os, sys, json
        PLUGIN_ROOT = Path(os.environ.get("LUNA_PLUGIN_ROOT", Path(__file__).resolve().parent / ".." / "luna-plugin")).resolve()
        if str(PLUGIN_ROOT) not in sys.path:
            sys.path.insert(0, str(PLUGIN_ROOT))

        PluginManager = None
        try:
            # luna-plugin의 PluginManager는 선택적입니다. 없으면 경고만 출력하고 계속 진행합니다.
            from sdk.manager import PluginManager  # type: ignore
        except Exception as e:
            print(f"[플러그인] 매니저 임포트 실패(sdk.manager): {e}")
            print(f"[플러그인] sys.path 일부: {sys.path[:5]}")
            print("[플러그인] luna-plugin의 PluginManager가 없거나 경로가 잘못되었습니다. 플러그인 로딩 단계를 건너뜁니다.")

        plugin_cfg_primary = PLUGIN_ROOT / "config" / "config.json"
        plugin_cfg_fallback = PLUGIN_ROOT / "config" / "config_example.json"
        plugins_path = PLUGIN_ROOT / "plugins"

        plugin_config = {}
        if plugin_cfg_primary.exists():
            cfg_path = plugin_cfg_primary
        elif plugin_cfg_fallback.exists():
            cfg_path = plugin_cfg_fallback
        else:
            cfg_path = None

        if cfg_path:
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    plugin_config = json.load(f)
                print(f"[플러그인] 설정 로드: {cfg_path}")
            except Exception as e:
                print(f"[플러그인] 설정 로드 실패({cfg_path.name}): {e} → 빈 설정으로 진행")

        # PluginManager가 존재하는 경우에만 플러그인 탐색/로딩을 시도합니다.
        if PluginManager is not None and plugins_path.exists():
            try:
                pm = PluginManager(str(plugins_path), plugin_config)
                discovered = pm.discover_plugins()
                wanted = plugin_config.get("plugins", discovered)

                print(f"[플러그인] 탐색됨: {discovered}")
                print(f"[플러그인] 로드 대상: {wanted}")

                activated = []
                for name in wanted:
                    inst = pm.load_plugin(name)
                    if inst:
                        try:
                            pm.enable_plugin(name)
                            activated.append(name)
                        except Exception as e:
                            print(f"[플러그인 활성화 실패] {name}: {e}")
                    else:
                        print(f"[플러그인 로딩 실패] {name}: (plugins/{name}/{name}_plugin.py & __init__.py 확인)")
                print(f"[플러그인] 활성화 완료: {activated}")
            except Exception as e:
                print(f"[플러그인] PluginManager 사용 중 오류: {e}")
        else:
            if not plugins_path.exists():
                print("[플러그인] 플러그인 디렉터리 없음 또는 PluginManager 미존재. 플러그인 미적용.")
        
        try:
            self.internal_mcp = LunaMCPInternal(
                name="LUNA-MCP", version="0.1.0",
                asr=asr_service, emotion=emotion_service, multi_intent=multi_intent_service,
                vision=vision_service, tts=tts_service, translator=translator_service,
                llm=llm_service, memory=memory_service, logger=logger
            )
            self.internal_mcp.mount_sse(app, "/mcp")
            print("[MCP] 내부 MCP 서버(SSE) 마운트: /mcp")
        except Exception as e:
            print(f"[MCP] 내부 MCP 서버 마운트 실패: {e}")

        try:
            self.mcp_mgr = ExternalMCPManager(config_path="config/mcp_servers.json", logger=logger)
            await self.mcp_mgr.start_enabled()
            print("[MCP] 외부 MCP 서버(ENABLED=true) 시작 완료")
            
            # MCPToolManager 초기화 및 도구 동기화
            self.tool_manager = MCPToolManager(self.mcp_mgr, tool_registry, logger=logger)
            await self.tool_manager.initialize()
            print("[MCP] MCP 도구 매니저 초기화 완료")
            
            # InteractionService에 MCPToolManager 설정
            self.interaction_service.mcp_tool_manager = self.tool_manager
            print("[MCP] InteractionService에 MCP 도구 매니저 연결 완료")
            
        except Exception as e:
            print(f"[MCP] 외부 MCP 매니저 시작 실패: {e}")

        # __aenter__는 반환값 없어도 됩니다.
        return

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if hasattr(self, "tool_manager") and self.tool_manager:
                print("[MCP] MCP 도구 매니저 정리 중...")
        except Exception:
            logger.exception("[MCP] 도구 매니저 정리 중 오류")
            
        try:
            if hasattr(self, "mcp_mgr") and self.mcp_mgr:
                await self.mcp_mgr.stop_all()
                print("[MCP] 외부 MCP 서버 종료 완료")
        except Exception:
            logger.exception("[MCP] 종료 중 오류")
            
        logger.info("[L.U.N.A. Shutdown] 서버 종료 중...")
        return False  # 예외 전파
    
_lifespan_ref: AppLifespan | None = None
def lifespan(app: FastAPI):
    global _lifespan_ref
    _lifespan_ref = AppLifespan()
    return _lifespan_ref
    
# FastAPI 앱 초기화
app = FastAPI(
    title="L.U.N.A. Core Agent API",
    description="L.U.N.A. Core Backend with MCP-based Agent",
    version="2.2.0-Agent",
    lifespan=lifespan
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",  # Vite 개발 서버
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Request / Response 스키마 정의
class TextRequest(BaseModel):
    text: str
    style: str | None = None
    style_weight: float | None = None

class AnalyzeResponse(BaseModel):
    emotion: dict[str, float]
    intents: dict[str, float]
    
class VisionResponse(BaseModel):
    answer: str

class SynthesizeResponse(BaseModel):
    audio_url: str
    
class TranslateRequest(BaseModel):
    text: str
    from_lang: str
    to_lang: str
    
class TranslateResponse(BaseModel):
    translated_text: str

class LLMRequest(BaseModel):
    input: str
    temperature: float = 0.85
    max_tokens: int = 64

class LLMResponse(BaseModel):
    content: str
    
# Interact 
class InteractRequest(BaseModel):
    input: str
    use_tools: bool = False  # MCP 도구 사용 여부

# ====================================================================
# MCP Tool 관련 Request/Response 스키마
# ====================================================================

class MCPToolCallRequest(BaseModel):
    """MCP 도구 호출 요청"""
    server_id: str  # "echo", "spotify" 등
    tool_name: str  # "ping", "play" 등
    arguments: dict  # 도구별 인자

class MCPToolCallResponse(BaseModel):
    """MCP 도구 호출 응답"""
    success: bool
    result: dict | Any
    error: str | None = None

class MCPToolInfo(BaseModel):
    """MCP 도구 정보"""
    id: str  # "echo/ping"
    name: str  # "echo/ping" (네임스페이스 포함)
    description: str
    inputSchema: dict

class MCPToolListResponse(BaseModel):
    """MCP 도구 목록 응답"""
    tools: list[MCPToolInfo]
    total: int

# Health Check Endpoint
@app.get("/health", tags=["Utility"])
def health() -> dict[str, str]:
    """
    서비스 상태 확인용 엔드포인트
    """
    return {"server": "L.U.N.A.", "version": "1.3.0", "status": "healthy"}

# ====================================================================
# Logs 엔드포인트
# ====================================================================

@app.get("/logs", tags=["Logs"])
def get_logs(limit: int = 500) -> List[dict]:
    """
    최근 로그를 반환합니다.
    
    Args:
        limit: 반환할 최근 로그 개수 (기본값: 500)
    
    Returns:
        List[dict]: 로그 항목 목록
    
    Example:
        GET /logs?limit=100
    """
    # TODO: 실제 로그 저장소에서 로그를 가져오세요
    # 현재는 빈 리스트를 반환합니다
    return []

# ====================================================================
# Settings 엔드포인트
# ====================================================================

class SettingsPayload(BaseModel):
    """설정 정보"""
    notifications: bool = True
    language: str = "ko"
    ttsSpeed: float = 1.0
    volume: int = 70
    experimentalFeatures: bool = False

@app.get("/settings", tags=["Settings"], response_model=SettingsPayload)
def get_settings() -> SettingsPayload:
    """
    현재 시스템 설정을 반환합니다.
    
    Returns:
        SettingsPayload: 설정 정보
    
    Example:
        GET /settings
    """
    # TODO: 실제 설정 저장소에서 설정을 가져오세요
    # 현재는 기본값을 반환합니다
    return SettingsPayload(
        notifications=True,
        language="ko",
        ttsSpeed=1.0,
        volume=70,
        experimentalFeatures=False
    )

@app.put("/settings", tags=["Settings"], response_model=SettingsPayload)
def update_settings(settings: SettingsPayload) -> SettingsPayload:
    """
    시스템 설정을 업데이트합니다.
    
    Args:
        settings: 새로운 설정 정보
    
    Returns:
        SettingsPayload: 업데이트된 설정
    
    Example:
        PUT /settings
        {
            "notifications": true,
            "language": "ko",
            "ttsSpeed": 1.0,
            "volume": 70,
            "experimentalFeatures": false
        }
    """
    # TODO: 실제 설정 저장소에 설정을 저장하세요
    # 현재는 받은 설정을 그대로 반환합니다
    logger.info(f"[Settings] 설정 업데이트: {settings}")
    return settings

# ====================================================================
# LLM Cache 엔드포인트
# ====================================================================

class LLMCacheStats(BaseModel):
    """LLM 캐시 통계"""
    cache_size: int = 0
    max_cache_size: int = 0
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0

@app.get("/llm/cache/stats", tags=["LLM Cache"], response_model=LLMCacheStats)
def get_llm_cache_stats() -> LLMCacheStats:
    """
    LLM 캐시 통계를 반환합니다.
    
    Returns:
        LLMCacheStats: 캐시 통계
    
    Example:
        GET /llm/cache/stats
    """
    # TODO: 실제 LLM 캐시 통계를 수집하세요
    # 현재는 기본값을 반환합니다
    return LLMCacheStats(
        cache_size=0,
        max_cache_size=0,
        hits=0,
        misses=0,
        total_requests=0,
        hit_rate=0.0
    )

# ====================================================================
# Metrics 엔드포인트
# ====================================================================

class SystemMetrics(BaseModel):
    """시스템 메트릭"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: float = 0.0
    temperature: float = 0.0
    uptime: int = 0
    active_connections: int = 0
    timestamp: Optional[str] = None

class APIStats(BaseModel):
    """API 통계"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    requests_per_second: float = 0.0

@app.get("/metrics/system", tags=["Metrics"], response_model=SystemMetrics)
def get_system_metrics() -> SystemMetrics:
    """
    시스템 메트릭을 반환합니다.
    
    Returns:
        SystemMetrics: CPU, 메모리, GPU, 온도, 가동 시간, 활성 연결 수
    
    Example:
        GET /metrics/system
    """
    import psutil
    from datetime import datetime
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
    except:
        cpu_percent = 0.0
        memory_percent = 0.0
    
    return SystemMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        gpu_percent=0.0,
        temperature=0.0,
        uptime=0,
        active_connections=0,
        timestamp=datetime.now().isoformat()
    )

@app.get("/metrics/api", tags=["Metrics"], response_model=APIStats)
def get_api_stats() -> APIStats:
    """
    API 통계를 반환합니다.
    
    Returns:
        APIStats: API 요청 통계
    
    Example:
        GET /metrics/api
    """
    # TODO: 실제 API 통계를 수집하세요
    # 현재는 기본값을 반환합니다
    return APIStats(
        total_requests=0,
        successful_requests=0,
        failed_requests=0,
        average_response_time=0.0,
        requests_per_second=0.0
    )

# ====================================================================
# MCP Tool 엔드포인트
# ====================================================================

@app.get("/mcp/tools", tags=["MCP"])
def get_mcp_tools() -> MCPToolListResponse:
    """
    모든 등록된 MCP 도구 목록을 반환합니다.
    
    Returns:
        MCPToolListResponse: 도구 목록
    
    Example:
        GET /mcp/tools
        
        Response:
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
    """
    if not hasattr(_lifespan_ref, "tool_manager") or _lifespan_ref.tool_manager is None:
        return MCPToolListResponse(tools=[], total=0)
    
    tool_list = _lifespan_ref.tool_manager.get_tool_list()
    mcp_tools = [MCPToolInfo(**tool) for tool in tool_list]
    
    return MCPToolListResponse(
        tools=mcp_tools,
        total=len(mcp_tools)
    )

@app.post("/mcp/call", tags=["MCP"])
async def call_mcp_tool(request: MCPToolCallRequest) -> MCPToolCallResponse:
    """
    MCP 도구를 호출합니다.
    
    Args:
        request: MCPToolCallRequest
            - server_id: MCP 서버 ID (e.g., "echo")
            - tool_name: 도구 이름 (e.g., "ping")
            - arguments: 도구에 전달할 인자 딕셔너리
    
    Returns:
        MCPToolCallResponse: 호출 결과
    
    Example:
        POST /mcp/call
        
        Request:
        {
            "server_id": "echo",
            "tool_name": "ping",
            "arguments": {
                "text": "Hello, world!"
            }
        }
        
        Response:
        {
            "success": true,
            "result": "Hello, world!",
            "error": null
        }
    """
    if not hasattr(_lifespan_ref, "tool_manager") or _lifespan_ref.tool_manager is None:
        return MCPToolCallResponse(
            success=False,
            result=None,
            error="MCP Tool Manager not initialized"
        )
    
    try:
        logger.info(f"[MCP] 도구 호출: {request.server_id}/{request.tool_name}")
        
        result = await _lifespan_ref.tool_manager.call_tool(
            server_id=request.server_id,
            tool_name=request.tool_name,
            arguments=request.arguments
        )
        
        return MCPToolCallResponse(
            success=True,
            result=result,
            error=None
        )
        
    except Exception as e:
        logger.error(f"[MCP] 도구 호출 실패: {e}")
        return MCPToolCallResponse(
            success=False,
            result=None,
            error=str(e)
        )

# ASR WebSocket Endpoint
@app.websocket("/ws/asr")
async def websocket_asr_endpoint(websocket: WebSocket):
    """
    ASR WebSocket 엔드포인트

    Args:
        websocket (WebSocket): WebSocket 연결 객체
    """
    await websocket.accept()
    
    async def result_callback(text: str):
        response_data = interaction_service.run(text)
        await websocket.send_json(response_data.model_dump())

    asr_stream = asr_service.transcribe_stream(result_callback)
    
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            asr_stream.write(audio_data)
    except WebSocketDisconnect:
        print("[L.U.N.A. WebSocket] 클라이언트 연결 종료")
    finally:
        asr_service.stop_transcription()
    
# Vision Analysis Endpoint
@app.post("/analyze/vision", response_model=VisionResponse, tags=["Analysis"])
async def analyze_vision(file: UploadFile = File(...)):
    """
    이미지를 입력받아 Vision 분석 결과를 반환합니다.

    Args:
        file (UploadFile): 분석할 이미지 파일
    Returns:
        AnalyzeResponse: 분석된 감정 및 인텐트 확률 분포
    """
    image_bytes = await file.read()
    try:
        answer = await vision_service.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[L.U.N.A. Vision] 분석 중 오류 발생: {str(e)}")

    return VisionResponse(
        answer=answer
    )

# Text-to-Speech Synthesis Endpoint
@app.post("/synthesize", response_model=SynthesizeResponse, tags=["Synthesis"])
async def synthesize_text(request: TextRequest):
    """
    텍스트를 입력받아 TTS 음성 합성 결과를 반환합니다.
    (비동기 처리로 성능 향상)

    Args:
        request (TextRequest): 합성할 텍스트
    Returns:
        JSONResponse: 합성된 음성 파일의 URL 및 스타일 정보
    """
    try:
        result: dict[str, Any] = await tts_service.synthesize_async(
            text=request.text,
            style=request.style,
            style_weight=request.style_weight
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[L.U.N.A. TTS] 합성 중 오류 발생: {str(e)}")
    
    return JSONResponse(
        content=result
    )

@app.post("/synthesize/parallel", response_model=SynthesizeResponse, tags=["Synthesis"])
async def synthesize_text_parallel(request: TextRequest):
    """
    텍스트를 입력받아 병렬 처리로 TTS 음성 합성 (긴 텍스트에 효과적)

    Args:
        request (TextRequest): 합성할 텍스트
    Returns:
        JSONResponse: 합성된 음성 파일의 URL 및 스타일 정보
    """
    try:
        result: dict[str, Any] = await tts_service.synthesize_parallel(
            text=request.text,
            style=request.style,
            style_weight=request.style_weight
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[L.U.N.A. TTS] 합성 중 오류 발생: {str(e)}")
    
    return JSONResponse(
        content=result
    )

# Text Translation Endpoint
@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
def translate_text(request: TranslateRequest):
    """
    텍스트 번역을 위한 엔드포인트

    Args:
        request (TranslateRequest): 번역할 텍스트 및 언어 정보
    Returns:
        TranslateResponse: 번역된 텍스트
    """
    try:
        translated = translator_service.translate(
            text=request.text,
            from_lang=request.from_lang,
            to_lang=request.to_lang
        )
        return TranslateResponse(translated_text=translated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[L.U.N.A. Translate] 번역 중 오류 발생: {str(e)}")
    
# LLM Generation Endpoint
@app.post("/generate", response_model=LLMResponse, tags=["LLM"])
def generate_text(request: LLMRequest):
    """
    LLM 서버에 요청을 보내고 응답을 반환합니다.

    Args:
        request (LLMRequest): LLM 요청 정보
    Returns:
        LLMResponse: LLM의 응답 텍스트
    """
    try:
        content = llm_service.generate(
            input_text=request.input,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return LLMResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[L.U.N.A. LLM] 생성 중 오류 발생: {str(e)}")
    
# Interact Endpoint
@app.post("/interact", response_model=InteractResponse, tags=["Interaction"])
def interact(request: InteractRequest):
    if not interaction_service:
        raise HTTPException(status_code=503, detail="서비스가 아직 준비되지 않았습니다.")
    try:
        return interaction_service.run(request.input, use_tools=request.use_tools)
    except Exception as e:
        logger.error(f"[Interact] 상호작용 중 심각한 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="상호작용 처리 중 서버 오류 발생")

# Streaming Interaction Endpoint (API 모드 전용)
@app.post("/interact/stream", tags=["Interaction"])
async def interact_stream(request: InteractRequest):
    """스트리밍 방식으로 LLM 응답을 실시간 전송합니다 (API 모드만 지원)."""
    if not interaction_service:
        raise HTTPException(status_code=503, detail="서비스가 아직 준비되지 않았습니다.")
    
    if not llm_service or llm_service.get_mode() != "api":
        raise HTTPException(
            status_code=400,
            detail="스트리밍은 API 모드에서만 지원됩니다. config/models.yaml에서 llm.mode를 'api'로 설정하세요."
        )
    
    try:
        
        async def stream_generator():
            # 1. 감정 분석
            emotion_result = interaction_service._analyze_emotion(request.input)
            yield f"data: {json.dumps({'type': 'emotion', 'data': emotion_result}, ensure_ascii=False)}\n\n"
            
            # 2. 번역 (필요시)
            translated = interaction_service._translate_if_needed(request.input)
            if translated != request.input:
                yield f"data: {json.dumps({'type': 'translation', 'data': translated}, ensure_ascii=False)}\n\n"
            
            # 3. 메모리 컨텍스트 로드
            messages = []
            if memory_service:
                messages = memory_service.get_context_for_llm()
                messages.append({"role": "user", "content": translated})
            
            # 4. LLM 스트리밍 응답
            target = list(llm_service.get_available_targets())[0]
            stream_response = llm_service.generate(
                target=target,
                messages=messages,
                stream=True
            )
            
            full_response = ""
            for chunk in stream_response:
                if "error" in chunk:
                    yield f"data: {json.dumps({'type': 'error', 'data': chunk['error']}, ensure_ascii=False)}\n\n"
                    break
                
                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'type': 'llm_chunk', 'data': content}, ensure_ascii=False)}\n\n"
                    
                    finish_reason = chunk["choices"][0].get("finish_reason")
                    if finish_reason == "stop":
                        # 5. 메모리 저장
                        if memory_service:
                            memory_service.add_entry(user_input=request.input, llm_response=full_response)
                        
                        yield f"data: {json.dumps({'type': 'complete', 'data': full_response}, ensure_ascii=False)}\n\n"
                        break
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
    except Exception as e:
        logger.error(f"[Interact Stream] 스트리밍 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"스트리밍 처리 중 오류 발생: {str(e)}")

# Memory Management Endpoints
@app.get("/memory/recent", tags=["Memory"])
def get_recent_memory(count: int = 10):
    """최근 대화 내역을 반환합니다."""
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    memory = memory_service.load_memory()
    return {"recent_conversations": memory[-count:] if memory else []}

@app.delete("/memory/clear", tags=["Memory"])
def clear_memory():
    """모든 대화 내역을 삭제합니다."""
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    memory_service.clear_memory()
    return {"status": "success", "message": "메모리가 삭제되었습니다."}

@app.post("/memory/summarize", tags=["Memory"])
def force_summarize_memory():
    """대화 내역을 수동으로 요약합니다."""
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    try:
        success = memory_service.force_summarize()
        if success:
            summary = memory_service.get_summary()
            return {
                "status": "success",
                "message": "대화 요약이 완료되었습니다.",
                "summary": summary
            }
        else:
            return {
                "status": "failed",
                "message": "요약할 대화가 충분하지 않습니다."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 중 오류 발생: {str(e)}")

@app.get("/memory/summary", tags=["Memory"])
def get_memory_summary():
    """현재 저장된 대화 요약을 반환합니다."""
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    summary = memory_service.get_summary()
    if summary:
        return {"summary": summary}
    else:
        return {"summary": None, "message": "저장된 요약이 없습니다."}

# Cache Management Endpoints
@app.get("/cache/stats", tags=["Cache"])
def get_cache_stats():
    """캐시 통계 정보를 반환합니다."""
    if not llm_service or not hasattr(llm_service.service, 'cache') or not llm_service.service.cache:
        return {"error": "캐시가 비활성화되어 있습니다."}
    return llm_service.service.cache.get_stats()

@app.post("/cache/cleanup", tags=["Cache"])
def cleanup_expired_cache():
    """만료된 캐시 항목을 정리합니다."""
    if not llm_service or not hasattr(llm_service.service, 'cache') or not llm_service.service.cache:
        return {"error": "캐시가 비활성화되어 있습니다."}
    
    removed = llm_service.service.cache.cleanup_expired()
    return {"status": "success", "removed": removed, "message": f"{removed}개의 만료된 캐시가 삭제되었습니다."}

@app.delete("/cache/clear", tags=["Cache"])
def clear_cache():
    """모든 캐시를 삭제합니다."""
    if not llm_service or not hasattr(llm_service.service, 'cache') or not llm_service.service.cache:
        return {"error": "캐시가 비활성화되어 있습니다."}
    
    llm_service.service.cache.clear()
    return {"status": "success", "message": "모든 캐시가 삭제되었습니다."}

# TTS Cache Management Endpoints
@app.get("/tts/cache/stats", tags=["TTS Cache"])
def get_tts_cache_stats():
    """TTS 캐시 통계 정보를 반환합니다."""
    if not tts_service or not tts_service.cache:
        return {"error": "TTS 캐시가 비활성화되어 있습니다."}
    return tts_service.cache.get_stats()

@app.post("/tts/cache/cleanup", tags=["TTS Cache"])
def cleanup_tts_cache():
    """만료된 TTS 캐시 항목을 정리합니다."""
    if not tts_service or not tts_service.cache:
        return {"error": "TTS 캐시가 비활성화되어 있습니다."}
    
    removed = tts_service.cache.cleanup_expired()
    return {
        "status": "success",
        "removed": removed,
        "message": f"{removed}개의 만료된 TTS 캐시가 삭제되었습니다."
    }

@app.delete("/tts/cache/clear", tags=["TTS Cache"])
def clear_tts_cache():
    """모든 TTS 캐시를 삭제합니다."""
    if not tts_service or not tts_service.cache:
        return {"error": "TTS 캐시가 비활성화되어 있습니다."}
    
    tts_service.cache.clear()
    return {
        "status": "success",
        "message": "모든 TTS 캐시가 삭제되었습니다."
    }

# ==================== Memory Management API ====================

@app.get("/memory/conversations", tags=["Memory"])
def get_conversations(
    user_id: str = "default",
    session_id: str = "default",
    limit: int = 50,
    offset: int = 0
):
    """
    대화 목록 조회
    
    Args:
        user_id: 사용자 ID
        session_id: 세션 ID
        limit: 최대 개수 (1-1000)
        offset: 시작 위치
        
    Returns:
        대화 목록
    """
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    from services.memory.database import get_db_manager
    from services.memory.repository import MemoryRepository
    
    db_manager = get_db_manager()
    repository = MemoryRepository(db_manager)
    
    conversations = repository.get_conversations(
        user_id=user_id,
        session_id=session_id,
        limit=min(limit, 1000),
        offset=offset
    )
    
    return {
        "conversations": [
            {
                "id": conv.id,
                "timestamp": conv.timestamp.isoformat(),
                "user_message": conv.user_message,
                "assistant_message": conv.assistant_message,
                "emotion": conv.emotion,
                "intent": conv.intent,
                "processing_time": conv.processing_time,
                "cached": conv.cached
            }
            for conv in conversations
        ],
        "count": len(conversations),
        "limit": limit,
        "offset": offset
    }

@app.get("/memory/conversations/{conversation_id}", tags=["Memory"])
def get_conversation(conversation_id: int):
    """
    특정 대화 조회
    
    Args:
        conversation_id: 대화 ID
        
    Returns:
        대화 상세 정보
    """
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    from services.memory.database import get_db_manager
    from services.memory.repository import MemoryRepository
    
    db_manager = get_db_manager()
    repository = MemoryRepository(db_manager)
    
    conversation = repository.get_conversation_by_id(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    
    return {
        "id": conversation.id,
        "user_id": conversation.user_id,
        "session_id": conversation.session_id,
        "timestamp": conversation.timestamp.isoformat(),
        "user_message": conversation.user_message,
        "assistant_message": conversation.assistant_message,
        "emotion": conversation.emotion,
        "intent": conversation.intent,
        "processing_time": conversation.processing_time,
        "cached": conversation.cached,
        "metadata": conversation.metadata,
        "created_at": conversation.created_at.isoformat()
    }

@app.post("/memory/conversations/search", tags=["Memory"])
def search_conversations(
    user_id: str | None = None,
    session_id: str | None = None,
    keyword: str | None = None,
    emotion: str | None = None,
    intent: str | None = None,
    limit: int = 50,
    offset: int = 0
):
    """
    대화 검색
    
    Args:
        user_id: 사용자 ID
        session_id: 세션 ID
        keyword: 검색 키워드 (메시지 내용)
        emotion: 감정 필터
        intent: 의도 필터
        limit: 최대 개수
        offset: 시작 위치
        
    Returns:
        검색 결과
    """
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    from services.memory.database import get_db_manager
    from services.memory.repository import MemoryRepository
    from services.memory.models import ConversationSearchRequest
    
    db_manager = get_db_manager()
    repository = MemoryRepository(db_manager)
    
    search_request = ConversationSearchRequest(
        user_id=user_id,
        session_id=session_id,
        keyword=keyword,
        emotion=emotion,
        intent=intent,
        limit=min(limit, 1000),
        offset=offset
    )
    
    results, total = repository.search_conversations(search_request)
    
    return {
        "results": [
            {
                "id": conv.id,
                "timestamp": conv.timestamp.isoformat(),
                "user_message": conv.user_message,
                "assistant_message": conv.assistant_message,
                "emotion": conv.emotion,
                "intent": conv.intent
            }
            for conv in results
        ],
        "total": total,
        "count": len(results),
        "limit": limit,
        "offset": offset
    }

@app.delete("/memory/conversations/{conversation_id}", tags=["Memory"])
def delete_conversation(conversation_id: int):
    """
    특정 대화 삭제
    
    Args:
        conversation_id: 대화 ID
        
    Returns:
        삭제 결과
    """
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    from services.memory.database import get_db_manager
    from services.memory.repository import MemoryRepository
    
    db_manager = get_db_manager()
    repository = MemoryRepository(db_manager)
    
    success = repository.delete_conversation(conversation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    
    return {
        "status": "success",
        "message": f"대화 {conversation_id}가 삭제되었습니다."
    }

@app.get("/memory/summaries", tags=["Memory"])
def get_summaries(
    user_id: str = "default",
    session_id: str = "default",
    limit: int = 10
):
    """
    요약 목록 조회
    
    Args:
        user_id: 사용자 ID
        session_id: 세션 ID
        limit: 최대 개수
        
    Returns:
        요약 목록
    """
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    from services.memory.database import get_db_manager
    from services.memory.repository import MemoryRepository
    
    db_manager = get_db_manager()
    repository = MemoryRepository(db_manager)
    
    summaries = repository.get_summaries(
        user_id=user_id,
        session_id=session_id,
        limit=limit
    )
    
    return {
        "summaries": [
            {
                "id": summary.id,
                "timestamp": summary.timestamp.isoformat(),
                "content": summary.content,
                "summarized_turns": summary.summarized_turns,
                "created_at": summary.created_at.isoformat()
            }
            for summary in summaries
        ],
        "count": len(summaries)
    }

@app.get("/memory/stats", tags=["Memory"])
def get_memory_stats(
    user_id: str | None = None,
    session_id: str | None = None
):
    """
    메모리 통계 조회
    
    Args:
        user_id: 사용자 ID (선택)
        session_id: 세션 ID (선택)
        
    Returns:
        메모리 통계 정보
    """
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    from services.memory.database import get_db_manager
    from services.memory.repository import MemoryRepository
    
    db_manager = get_db_manager()
    repository = MemoryRepository(db_manager)
    
    stats = repository.get_stats(user_id=user_id, session_id=session_id)
    
    return {
        "total_conversations": stats.total_conversations,
        "total_summaries": stats.total_summaries,
        "unique_users": stats.unique_users,
        "unique_sessions": stats.unique_sessions,
        "first_conversation": stats.first_conversation.isoformat() if stats.first_conversation else None,
        "last_conversation": stats.last_conversation.isoformat() if stats.last_conversation else None,
        "emotions_distribution": stats.emotions_distribution,
        "intents_distribution": stats.intents_distribution,
        "avg_processing_time": stats.avg_processing_time,
        "cache_hit_rate": stats.cache_hit_rate,
        "conversations_by_date": stats.conversations_by_date
    }

@app.delete("/memory/clear", tags=["Memory"])
def clear_memory(
    user_id: str = "default",
    session_id: str = "default",
    confirm: bool = False
):
    """
    메모리 삭제 (주의!)
    
    Args:
        user_id: 사용자 ID
        session_id: 세션 ID
        confirm: 삭제 확인 (true 필수)
        
    Returns:
        삭제 결과
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="삭제를 확인하려면 confirm=true를 전달해야 합니다."
        )
    
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    
    from services.memory.database import get_db_manager
    from services.memory.repository import MemoryRepository
    
    db_manager = get_db_manager()
    repository = MemoryRepository(db_manager)
    
    deleted = repository.delete_conversations(
        user_id=user_id,
        session_id=session_id
    )
    
    return {
        "status": "success",
        "deleted_conversations": deleted,
        "message": f"{deleted}개의 대화가 삭제되었습니다."
    }

# MCP Endpoints
@app.get("/mcp/external/servers", tags=["MCP"])
def mcp_list_servers():
    mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    if not mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    return {"servers": [c.model_dump() for c in mgr.list_configs()]}

@app.post("/mcp/external/reload", tags=["MCP"])
async def mcp_reload_servers():
    mcp_mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    tool_mgr = getattr(_lifespan_ref, "tool_manager", None)
    
    if not mcp_mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    
    # ExternalMCPManager 재로드
    await mcp_mgr.reload_and_apply()
    
    # MCPToolManager 재로드 (도구 목록 재발견)
    if tool_mgr:
        await tool_mgr.reload()
    
    return {"status": "ok"}

@app.get("/mcp/external/{server_id}/tools", tags=["MCP"])
async def mcp_list_tools(server_id: str):
    mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    if not mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    tools = await mgr.list_tools(server_id)
    return {"server": server_id, "tools": [t.model_dump() for t in tools]}

@app.get("/mcp/external/{server_id}/resources", tags=["MCP"])
async def mcp_list_resources(server_id: str):
    mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    if not mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    resources = await mgr.list_resources(server_id)
    return {"server": server_id, "resources": [r.model_dump() for r in resources]}

class ToolCallPayload(BaseModel):
    name: str
    arguments: dict = {}

@app.post("/mcp/external/{server_id}/call", tags=["MCP"])
async def mcp_call_tool(server_id: str, payload: ToolCallPayload):
    mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    if not mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    result = await mgr.call_tool(server_id, payload.name, payload.arguments)
    # result는 MCP SDK의 통합 결과 모델. 그대로 반환(프론트에서 파싱)
    return {"server": server_id, "result": result}


# ====================================================================
#  MCP 설정 관리 엔드포인트 (실시간 추가/제거/활성화)
# ====================================================================

class MCPServerConfig(BaseModel):
    """MCP 서버 설정"""
    id: str
    transport: str = "stdio"
    command: str = None  # 업데이트 시 선택적
    args: list = []
    cwd: str = None
    enabled: bool = True
    timeoutMs: int = 8000
    env: dict = {}
    
    class Config:
        extra = "allow"  # 추가 필드 허용

@app.post("/mcp/external/config/add", tags=["MCP Config"])
async def mcp_add_server(config: MCPServerConfig):
    """
    새로운 MCP 서버를 설정에 추가하고 활성화합니다.
    
    Example:
    ```json
    {
        "id": "my-plugin",
        "command": "python",
        "args": ["server.py"],
        "cwd": "/path/to/plugin",
        "enabled": true,
        "env": {}
    }
    ```
    """
    mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    if not mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    
    try:
        # 설정 파일 로드
        config_path = Path("config/mcp_servers.json")
        
        data = {}
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
        
        mcp_servers = data.get("mcpServers", {})
        
        # 중복 체크
        if config.id in mcp_servers:
            raise HTTPException(400, f"서버 '{config.id}' 이미 존재")
        
        # 새 서버 추가
        mcp_servers[config.id] = config.model_dump(exclude_unset=True)
        data["mcpServers"] = mcp_servers
        
        # 설정 저장
        config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # 설정 재로드
        await mgr.reload_and_apply()
        
        # 도구 목록 재로드
        tool_mgr = getattr(_lifespan_ref, "tool_manager", None)
        if tool_mgr:
            await tool_mgr.reload()
        
        logger.info(f"[MCP Config] 서버 추가됨: {config.id}")
        return {"status": "ok", "server": config.id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MCP Config] 서버 추가 실패: {e}")
        raise HTTPException(500, str(e))


@app.post("/mcp/external/config/remove", tags=["MCP Config"])
async def mcp_remove_server(server_id: str):
    """
    MCP 서버를 설정에서 제거합니다.
    
    Example: POST /mcp/external/config/remove?server_id=my-plugin
    """
    mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    if not mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    
    try:
        # 설정 파일 로드
        config_path = Path("config/mcp_servers.json")
        
        if not config_path.exists():
            raise HTTPException(404, "설정 파일 없음")
        
        data = json.loads(config_path.read_text(encoding="utf-8"))
        mcp_servers = data.get("mcpServers", {})
        
        # 해당 서버 확인
        if server_id not in mcp_servers:
            raise HTTPException(404, f"서버 '{server_id}' 없음")
        
        # 제거
        del mcp_servers[server_id]
        data["mcpServers"] = mcp_servers
        
        # 설정 저장
        config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # 설정 재로드
        await mgr.reload_and_apply()
        
        # 도구 목록 재로드
        tool_mgr = getattr(_lifespan_ref, "tool_manager", None)
        if tool_mgr:
            await tool_mgr.reload()
        
        logger.info(f"[MCP Config] 서버 제거됨: {server_id}")
        return {"status": "ok", "server": server_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MCP Config] 서버 제거 실패: {e}")
        raise HTTPException(500, str(e))


@app.post("/mcp/external/config/update", tags=["MCP Config"])
async def mcp_update_server(config: MCPServerConfig):
    """
    MCP 서버 설정을 업데이트합니다 (활성화/비활성화 포함).
    
    Example:
    ```json
    {
        "id": "my-plugin",
        "enabled": false
    }
    ```
    """
    mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    if not mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    
    try:
        logger.info(f"[MCP Config] 업데이트 요청: id={config.id}, 원본={config.dict()}")
        # 설정 파일 로드
        config_path = Path("config/mcp_servers.json")
        
        if not config_path.exists():
            raise HTTPException(404, "설정 파일 없음")
        
        data = json.loads(config_path.read_text(encoding="utf-8"))
        mcp_servers = data.get("mcpServers", {})
        
        # 해당 서버 확인
        if config.id not in mcp_servers:
            raise HTTPException(404, f"서버 '{config.id}' 없음")
        
        # 기존 설정과 병합
        existing = mcp_servers[config.id]
        update_data = config.model_dump(exclude_unset=True)  # exclude_none 대신 exclude_unset 사용
        logger.debug(f"[MCP Config] 업데이트 요청: id={config.id}, 변경 데이터={update_data}")
        existing.update(update_data)
        mcp_servers[config.id] = existing
        logger.debug(f"[MCP Config] 업데이트 후: id={config.id}, enabled={existing.get('enabled')}")
        data["mcpServers"] = mcp_servers
        
        # 설정 저장
        config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # 설정 재로드
        await mgr.reload_and_apply()
        
        # 도구 목록 재로드
        tool_mgr = getattr(_lifespan_ref, "tool_manager", None)
        if tool_mgr:
            await tool_mgr.reload()
        
        logger.info(f"[MCP Config] 서버 업데이트됨: {config.id}")
        return {"status": "ok", "server": config.id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MCP Config] 서버 업데이트 실패: {e}")
        raise HTTPException(500, str(e))


# ====================================================================
#  MCP URL 기반 로드 (웹에서 공유되는 MCP 파일 직접 로드)
# ====================================================================

class MCPURLConfig(BaseModel):
    """URL 기반 MCP 서버 설정"""
    id: str
    url: str
    namespace: str = None
    enabled: bool = True
    timeoutMs: int = 8000

@app.post("/mcp/external/config/add-from-url", tags=["MCP Config"])
async def mcp_add_server_from_url(config: MCPURLConfig):
    """
    웹에서 공유되는 MCP 파일을 URL로 직접 로드합니다.
    
    Example:
    ```json
    {
        "id": "my-plugin",
        "url": "https://raw.githubusercontent.com/user/repo/main/server.py",
        "namespace": "my-plugin",
        "enabled": true
    }
    ```
    
    지원하는 포맷:
    - Python 파일 (.py) - FastMCP 기반
    - npm 패키지 (package.json이 있는 경우)
    """
    mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    if not mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    
    try:
        # URL 유효성 확인
        if not config.url.startswith(("http://", "https://")):
            raise HTTPException(400, "유효한 URL을 입력하세요 (http:// 또는 https://)")
        
        # 임시 디렉토리 생성
        temp_dir = Path(tempfile.mkdtemp(prefix=f"mcp_{config.id}_"))
        
        logger.info(f"[MCP URL] 다운로드 시작: {config.url}")
        
        # URL에서 파일 다운로드
        try:
            urllib.request.urlretrieve(config.url, temp_dir / "server.py")
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise HTTPException(400, f"파일 다운로드 실패: {e}")
        
        # 설정 파일에 추가
        config_path = Path("config/mcp_servers.json")
        data = {}
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
        
        servers = data.get("servers", [])
        
        # 중복 체크
        if any(s["id"] == config.id for s in servers):
            shutil.rmtree(temp_dir)
            raise HTTPException(400, f"서버 '{config.id}' 이미 존재")
        
        # 새 서버 추가
        new_server = {
            "id": config.id,
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
            "cwd": str(temp_dir),
            "enabled": config.enabled,
            "namespace": config.namespace or config.id,
            "timeoutMs": config.timeoutMs,
            "source": {
                "type": "url",
                "url": config.url,
                "downloaded_at": str(Path(__file__).resolve().parent / "main.py")  # 타임스탬프 용도
            }
        }
        
        servers.append(new_server)
        data["servers"] = servers
        
        # 설정 저장
        config_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        
        # 설정 재로드
        await mgr.reload_and_apply()
        
        # 도구 목록 재로드
        tool_mgr = getattr(_lifespan_ref, "tool_manager", None)
        if tool_mgr:
            await tool_mgr.reload()
        
        logger.info(f"[MCP URL] 서버 추가됨: {config.id} (temp: {temp_dir})")
        return {
            "status": "ok",
            "server": config.id,
            "path": str(temp_dir),
            "url": config.url
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MCP URL] 서버 추가 실패: {e}")
        raise HTTPException(500, str(e))


class MCPConfigFileUpload(BaseModel):
    """Claude 포맷 MCP 설정 파일"""
    mcpServers: dict = {}

@app.post("/mcp/external/config/import-claude-format", tags=["MCP Config"])
async def mcp_import_claude_format(config_file: MCPConfigFileUpload):
    """
    Claude Desktop 포맷의 MCP 설정을 가져옵니다.
    
    Example:
    ```json
    {
        "mcpServers": {
            "google-maps": {
                "command": "docker",
                "args": ["run", "-i", "--rm", "-e", "GOOGLE_MAPS_API_KEY", "mcp/google-maps"],
                "env": {
                    "GOOGLE_MAPS_API_KEY": "your-key"
                }
            },
            "my-python-tool": {
                "command": "python",
                "args": ["server.py"],
                "cwd": "/path/to/server"
            }
        }
    }
    ```
    """
    mgr = getattr(_lifespan_ref, "mcp_mgr", None)
    if not mgr:
        raise HTTPException(503, "MCP 매니저 준비 안됨")
    
    try:
        config_path = Path("config/mcp_servers.json")
        data = {}
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
        
        servers = data.get("servers", [])
        added_count = 0
        
        # mcpServers의 각 서버를 변환
        for server_id, server_config in config_file.mcpServers.items():
            # 중복 체크
            if any(s["id"] == server_id for s in servers):
                logger.warning(f"[MCP Import] 서버 '{server_id}' 이미 존재, 스킵")
                continue
            
            # Claude 포맷을 L.U.N.A 포맷으로 변환
            new_server = {
                "id": server_id,
                "transport": "stdio",
                "command": server_config.get("command"),
                "args": server_config.get("args", []),
                "cwd": server_config.get("cwd"),
                "env": server_config.get("env", {}),
                "enabled": server_config.get("enabled", True),
                "namespace": server_config.get("namespace", server_id),
                "timeoutMs": server_config.get("timeoutMs", 8000),
                "source": {
                    "type": "claude-format",
                    "imported_at": Path(__file__).resolve().parent.name
                }
            }
            
            servers.append(new_server)
            added_count += 1
            logger.info(f"[MCP Import] 서버 추가됨: {server_id}")
        
        if added_count == 0:
            return {"status": "ok", "added": 0, "message": "추가할 새 서버가 없습니다"}
        
        # 설정 저장
        data["servers"] = servers
        config_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        
        # 설정 재로드
        await mgr.reload_and_apply()
        
        # 도구 목록 재로드
        tool_mgr = getattr(_lifespan_ref, "tool_manager", None)
        if tool_mgr:
            await tool_mgr.reload()
        
        logger.info(f"[MCP Import] 총 {added_count}개 서버 추가됨")
        return {
            "status": "ok",
            "added": added_count,
            "servers": list(config_file.mcpServers.keys())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MCP Import] 설정 가져오기 실패: {e}")
        raise HTTPException(500, str(e))


# ====================================================================
#  LLM + MCP 도구 연동
# ====================================================================

class LLMChatWithToolsRequest(BaseModel):
    """LLM 채팅 (도구 지원) 요청"""
    user_input: str
    llm_provider: str = "gemini"  # "gemini", "claude", 등
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    use_tools: bool = True
    tool_choice: str = "auto"  # "auto", "none", "required"

class LLMToolUse(BaseModel):
    """LLM이 호출한 도구"""
    tool_id: str  # "echo/ping" 형식
    server_id: str
    tool_name: str
    arguments: dict
    result: Optional[Any] = None
    error: Optional[str] = None

class LLMChatWithToolsResponse(BaseModel):
    """LLM 채팅 (도구 지원) 응답"""
    success: bool
    content: str  # 최종 응답
    tool_calls: List[LLMToolUse] = []
    usage: Optional[dict] = None

@app.post("/llm/chat-with-tools", tags=["LLM + MCP"])
async def llm_chat_with_tools(request: LLMChatWithToolsRequest) -> LLMChatWithToolsResponse:
    """
    LLM이 MCP 도구를 사용할 수 있는 채팅 엔드포인트입니다.
    
    LLM이 도구를 호출하면 자동으로 실행하고 결과를 다시 LLM에 전달합니다.
    
    Example:
    ```json
    {
        "user_input": "echo/ping 도구를 사용해서 'Hello'를 전송해주고, 그 결과를 알려줘",
        "llm_provider": "gemini",
        "use_tools": true,
        "tool_choice": "auto"
    }
    ```
    """
    llm_svc = getattr(_lifespan_ref, "llm_service", None)
    tool_mgr = getattr(_lifespan_ref, "tool_manager", None)
    
    if not llm_svc:
        raise HTTPException(503, "LLM 서비스 준비 안됨")
    
    if not tool_mgr or not request.use_tools:
        # 도구 없이 일반 채팅
        try:
            response = llm_svc.generate(
                target=request.llm_provider,
                user_prompt=request.user_input,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            if isinstance(response, dict):
                content = response.get("response", str(response))
            else:
                content = str(response)
            
            return LLMChatWithToolsResponse(
                success=True,
                content=content,
                tool_calls=[]
            )
        except Exception as e:
            logger.error(f"[LLM Chat] 생성 실패: {e}")
            raise HTTPException(500, str(e))
    
    # 도구 사용 모드
    try:
        tool_list = tool_mgr.get_tool_list()
        tools_schema = []
        
        for tool in tool_list:
            tools_schema.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool.get("inputSchema", {})
            })
        
        system_prompt = request.system_prompt or """You are a helpful assistant with access to tools. 
When the user asks you to use a tool, call it using the provided tool schema.
Always use tool calls when appropriate to help the user."""
        
        # LLM에 도구 정보와 함께 요청
        logger.info(f"[LLM Chat] 도구 사용 모드: {len(tools_schema)}개 도구 제공")
        
        # Gemini API의 경우 tool_choice 및 tools 파라미터 사용
        # 현재는 간단히 프롬프트에 도구 정보를 포함
        tool_info_str = "Available tools:\n"
        for tool in tools_schema:
            tool_info_str += f"- {tool['name']}: {tool['description']}\n"
        
        enhanced_prompt = f"{request.user_input}\n\n{tool_info_str}"
        
        response = llm_svc.generate(
            target=request.llm_provider,
            user_prompt=enhanced_prompt,
            system_prompt=system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if isinstance(response, dict):
            content = response.get("response", str(response))
        else:
            content = str(response)
        
        # 응답에서 도구 호출 추출 및 실행
        tool_calls = []
        
        # 간단한 도구 호출 패턴 인식 (예: "사용 도구: echo/ping with argument: 'test'")
        # 더 정교한 파싱은 LLM 응답 형식에 따라 구현 필요
        import re
        tool_pattern = r"([\w-]+/[\w-]+)\s*\(?([^)]*)\)?"
        matches = re.findall(tool_pattern, content)
        
        for match in matches:
            tool_id = match[0]
            args_str = match[1]
            
            # tool_id에서 server_id와 tool_name 추출
            if "/" in tool_id:
                server_id, tool_name = tool_id.split("/", 1)
                
                # 도구 실행
                try:
                    # 인자 파싱 (간단한 방식)
                    arguments = {}
                    if args_str:
                        # JSON 형식 시도
                        try:
                            arguments = json.loads(args_str)
                        except:
                            # 간단한 key=value 파싱
                            for pair in args_str.split(","):
                                if "=" in pair:
                                    k, v = pair.split("=", 1)
                                    arguments[k.strip()] = v.strip().strip("'\"")
                    
                    result = await tool_mgr.call_tool(server_id, tool_name, arguments)
                    
                    tool_call = LLMToolUse(
                        tool_id=tool_id,
                        server_id=server_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        result=str(result)
                    )
                    tool_calls.append(tool_call)
                    logger.info(f"[LLM Chat] 도구 호출 성공: {tool_id}")
                except Exception as e:
                    tool_call = LLMToolUse(
                        tool_id=tool_id,
                        server_id=server_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        error=str(e)
                    )
                    tool_calls.append(tool_call)
                    logger.error(f"[LLM Chat] 도구 호출 실패: {tool_id} - {e}")
        
        return LLMChatWithToolsResponse(
            success=True,
            content=content,
            tool_calls=tool_calls
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LLM Chat] 도구 사용 모드 실패: {e}")
        raise HTTPException(500, str(e))