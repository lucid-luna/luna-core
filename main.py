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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 서비스 모듈 import (추후 구현)
from services.emotion import EmotionService
from services.multi_intent import MultiIntentService
# from services.tts import TTSService

# FastAPI 앱 초기화
app = FastAPI(
    title="L.U.N.A. Core API",
    description="L.U.N.A. Core API",
    version="0.1.0",
)

# 서비스 인스턴스 생성
emotion_service = EmotionService()
multi_intent_service = MultiIntentService()

# Request / Response 스키마 정의
class TextRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    emotion: dict[str, float]
    intents: dict[str, float]

class SynthesizeResponse(BaseModel):
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
    
# Text-to-Speech Synthesis Endpoint
@app.post("/synthesize", response_model=SynthesizeResponse, tags=["Synthesis"])
def synthesize_text(request: TextRequest):
    """
    텍스트를 입력받아 TTS 음성 합성 결과를 반환합니다.

    Args:
        request (TextRequest): 합성할 텍스트
    Returns:
        SynthesizeResponse: 합성된 음성 파일 URL
    """
    # TODO: 실제 TTS 음성 합성 로직 구현

    # 예시 응답
    audio_url = "https://example.com/audio/12345"
    return SynthesizeResponse(
        audio_url=audio_url
    )