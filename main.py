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

from typing import Any

import logging
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import asynccontextmanager
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
from services.mcp.spotify import SpotifyService
from services.mcp.tool_registry import ToolRegistry
from services.memory import MemoryService
from services.interaction import InteractionService, InteractResponse
from utils.llm_config import create_llm_manager

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모든 서비스 초기화 및 웜업"""
    global asr_service, emotion_service, multi_intent_service, vision_service, tts_service, translator_service, llm_service, memory_service, interaction_service
    
    print("[L.U.N.A. Startup] 서비스 초기화 시작...")

    asr_service = ASRService()
    emotion_service = EmotionService()
    multi_intent_service = MultiIntentService()
    vision_service = VisionService()
    translator_service = TranslatorService()
    
    print("[L.U.N.A. Startup] LLM 서비스 초기화 중...")
    llm_service, llm_target = create_llm_manager(config_path="config/models.yaml")
    
    if llm_service is None:
        print("[L.U.N.A. Startup] 경고: LLM 서비스 초기화 실패. 기본 서버 모드로 대체합니다.")
        llm_server_configs = {
            "rp": {"url": "http://localhost:8080", "alias": "Luna"},
            "translator": {"url": "http://localhost:8081", "alias": "Translator"}
        }
        from services.llm import LLMService
        llm_service = LLMService(server_configs=llm_server_configs)
        llm_target = "rp"
    
    print(f"[L.U.N.A. Startup] LLM 모드: {llm_service.get_mode()}, 타겟: {llm_target}")
    
    print("[L.U.N.A. Startup] 메모리 서비스 초기화 중...")
    
    # 설정 파일에서 메모리 설정 로드
    config_path = Path("config/models.yaml")
    memory_config = {
        "max_entries": 50,
        "max_context_turns": 6,
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
        summary_threshold=memory_config["summary_threshold"],
        enable_auto_summary=memory_config["enable_auto_summary"],
        llm_service=llm_service  # LLM 서비스 전달
    )
    print(f"[L.U.N.A. Startup] 메모리 서비스 초기화 완료 (저장된 대화: {len(memory_service.load_memory())}개)")
    
    print("[L.U.N.A. Startup] TTS 서비스 초기화 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_service = TTSService(device=device)
    
    print("[L.U.N.A. Startup] TTS 웜업 시작...")
    try:
        warmup_result = tts_service.synthesize(
            text="テスト",
            style="Neutral",
            style_weight=1.0
        )
        print("[L.U.N.A. Startup] TTS 웜업 완료!")
    except Exception as e:
        print(f"[L.U.N.A. Startup] TTS 웜업 실패: {e}")
        
    spotify_service = SpotifyService(logger=logger)
    tool_registry = ToolRegistry(spotify_service=spotify_service)
        
    interaction_service = InteractionService(
        emotion_service=emotion_service,
        multi_intent_service=multi_intent_service,
        translator_service=translator_service,
        llm_service=llm_service,
        tts_service=tts_service,
        tool_registry=tool_registry,
        memory_service=memory_service,
        llm_target=llm_target,
        logger=logger
    )
    
    logger.info("[L.U.N.A. Startup] 모든 서비스 초기화 완료!")
    print("[L.U.N.A. Startup] 모든 서비스 초기화 완료!")
    yield
    logger.info("[L.U.N.A. Shutdown] 서버 종료 중...")
    
# FastAPI 앱 초기화
app = FastAPI(
    title="L.U.N.A. Core Agent API",
    description="L.U.N.A. Core Backend with MCP-based Agent",
    version="2.2.0-Agent",
    lifespan=lifespan
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
    
# Health Check Endpoint
@app.get("/health", tags=["Utility"])
def health() -> dict[str, str]:
    """
    서비스 상태 확인용 엔드포인트
    """
    return {"server": "L.U.N.A.", "version": "1.3.0", "status": "healthy"}

@app.get("/spotify/callback", tags=["Spotify"])
def spotify_callback(code: str = None, error: str = None):
    """
    Spotify OAuth 콜백 엔드포인트
    """
    if error:
        logger.error(f"[Spotify Callback] 인증 오류: {error}")
        return {"status": "error", "message": f"스포티파이 인증 오류: {error}"}
    
    logger.info(f"[Spotify Callback] 인증 코드 수신: {code}, 토큰 파일이 .spotify_cache에 저장됩니다.")
    return {"status": "ok", "message": "스포티파이 인증이 완료되었습니다. 이제 도구를 사용할 수 있습니다."}

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
def synthesize_text(request: TextRequest):
    """
    텍스트를 입력받아 TTS 음성 합성 결과를 반환합니다.

    Args:
        request (TextRequest): 합성할 텍스트
    Returns:
        JSONResponse: 합성된 음성 파일의 URL 및 스타일 정보
    """
    try:
        result: dict[str, Any] = tts_service.synthesize(
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
        return interaction_service.run(request.input)
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
        import json
        
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
@app.get("/memory/stats", tags=["Memory"])
def get_memory_stats():
    """메모리 통계 정보를 반환합니다."""
    if not memory_service:
        raise HTTPException(status_code=503, detail="메모리 서비스가 준비되지 않았습니다.")
    return memory_service.get_memory_stats()

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
