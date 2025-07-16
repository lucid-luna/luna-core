# ====================================================================
#  File: main.py
# ====================================================================
"""
L.U.N.A. Core Backend Entry Point

• FastAPI 앱을 초기화하고, 주요 분석 및 합성 API 엔드포인트를 정의합니다.
• /health: 서비스 상태 확인
• /analyze/text: 텍스트 입력 → Emotion & Intent 분석
• /synthesize: 텍스트 입력 → TTS 음성 합성 호출

실행 예시:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.params import File
from pydantic import BaseModel
import torch

from services.emotion import EmotionService
from services.multi_intent import MultiIntentService
from services.vision import VisionService
from services.tts import TTSService
from services.translator import TranslatorService
from services.llm import LLMService

from utils.osc_sender import OSCSender

# FastAPI 앱 초기화
app = FastAPI(
    title="L.U.N.A. Core API",
    description="L.U.N.A. Core API",
    version="1.1.0",
)

# 서비스 인스턴스 생성
emotion_service = EmotionService()
multi_intent_service = MultiIntentService()
vision_service = VisionService()
tts_service = TTSService(device="cuda" if torch.cuda.is_available() else "cpu")
translator_service = TranslatorService()
llm_service = LLMService(server_url="http://localhost:8080")

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
    temperature: float = 0.7
    max_tokens: int = 256

class LLMResponse(BaseModel):
    content: str
    
# Interact 
class InteractRequest(BaseModel):
    input: str
    
class InteractResponse(BaseModel):
    text: str
    emotion: str
    intent: str
    style: str
    audio_url: str
    
# Health Check Endpoint
@app.get("/health", tags=["Utility"])
def health() -> dict[str, str]:
    """
    서비스 상태 확인용 엔드포인트
    """
    return {"status": "ok"}

# Text Analysis Endpoint
@app.post("/analyze/text", response_model=AnalyzeResponse, tags=["Analysis"])
def analyze_text(request: TextRequest):
    """
    텍스트를 입력받아 Emotion & Intent 분석 결과를 반환합니다.

    Args:
        request (TextRequest): 분석할 텍스트
    Returns:
        AnalyzeResponse: 분석된 감정 및 인텐트 확률 분포
    """
    emotions = emotion_service.predict(request.text)
    intents = multi_intent_service.predict(request.text)
    return AnalyzeResponse(
        emotion=emotions,
        intents=intents
    )
    
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
    """
    전체 모델의 상호작용을 처리하는 엔드포인트
    """
    try:
        print("[L.U.N.A. Interact] 입력 텍스트:", request.input)
        
        ko_input = translator_service.translate(
            text=request.input,
            from_lang="ko",
            to_lang="en"
        )
        
        print("[L.U.N.A. Interact] LLM Input (EN):", ko_input)
        
        emotion_probs = emotion_service.predict(ko_input)
        intent_probs = multi_intent_service.predict(ko_input)
        top_emotion = max(emotion_probs, key=emotion_probs.get) if emotion_probs else "neutral"
        top_intent = max(intent_probs, key=intent_probs.get) if intent_probs else "greeting"

        osc = OSCSender()
        osc.send_emotion(top_emotion)

        print("[L.U.N.A. Interact] Emotion:", top_emotion, "Intent:", top_intent)
        
        from utils.style_map import get_style_from_emotion
        style, style_weight = get_style_from_emotion(top_emotion)
        print("[L.U.N.A. Interact] Style:", style, "Weight:", style_weight)
        
        en_response = llm_service.generate(
            input_text=ko_input,
            temperature=0.85,
            max_tokens=256
        )
        
        print("[L.U.N.A. Interact] LLM Response (EN):", en_response)
        
        jp_response = translator_service.translate(
            text=en_response,
            from_lang="en",
            to_lang="ja"
        )
        
        print("[L.U.N.A. Interact] TTS Input (JP):", jp_response)
        
        result: dict[str, Any] = tts_service.synthesize(
            text=jp_response,
            style=style,
            style_weight=style_weight
        )
        
        return InteractResponse(
            text=jp_response,
            emotion=top_emotion,
            intent=top_intent,
            style=style,
            audio_url=result["audio_url"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[L.U.N.A. Interact] 상호작용 중 오류 발생: {str(e)}")