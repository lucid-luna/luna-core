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
from utils.style_map import get_style_from_emotion, get_top_emotion

# FastAPI 앱 초기화
app = FastAPI(
    title="L.U.N.A. Core API",
    description="L.U.N.A. Core API",
    version="1.0.0",
)

# 서비스 인스턴스 생성
emotion_service = EmotionService()
multi_intent_service = MultiIntentService()
vision_service = VisionService()
tts_service = TTSService(device="cuda" if torch.cuda.is_available() else "cpu")
translator_service = TranslatorService()

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