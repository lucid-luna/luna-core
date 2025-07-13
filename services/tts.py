# ====================================================================
#  File: services/tts.py
# ====================================================================
"""
L.U.N.A. TTS 합성 서비스 모듈

 - luna_core.tts_model.TTSModelHolder를 사용하여 실제 음성 합성을 실행
 - 합성된 오디오는 outputs 디렉토리에 WAV로 저장되고, 그 URL을 반환
 
    2025/07/13
     - 예외처리 추가 및 전체 코드 옵티마이징
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, List

import torch
import soundfile as sf
from fastapi import HTTPException
from utils.config import load_config_dict
from utils.style_map import get_style_from_emotion, get_top_emotion
from services.emotion import EmotionService
from models.tts_model import TTSModelHolder


# ─────────────────────────────────────────────────────────────────────
# 예외 처리
# ─────────────────────────────────────────────────────────────────────
class TTSError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code, self.detail = code, detail

# ─────────────────────────────────────────────────────────────────────
# TTSService 클래스
# ─────────────────────────────────────────────────────────────────────
class TTSService:
    """
    L.U.N.A. TTS 합성 서비스 클래스
    
    텍스트 -> 음성 합성
    """
    
    _holder: Optional[TTSModelHolder] = None
    _model: Optional[torch.nn.Module] = None
    _sampling_rate: Optional[int] = None
    _config: Optional[dict] = None

    def __init__(self, device: str):
        tts_config = load_config_dict("models")["tts"]        
        self.device = device
        
        model_dir = Path(tts_config["model_dir"]).expanduser()
        self.default_name = tts_config.get("default_model")
        self.outputs_dir  = Path(tts_config.get("output_dir", "outputs"))
        self.outputs_dir.mkdir(exist_ok=True)
        
        if TTSService._holder is None or TTSService._config != tts_config:
            TTSService._config = tts_config
            TTSService._holder = TTSModelHolder(str(model_dir), device)
            self._load_default_model(tts_config)

        TTSService._sampling_rate = tts_config.get("sampling_rate", 44100)
        self.noise_scale   = tts_config.get("noise_scale", 0.6)
        self.noise_scale_w = tts_config.get("noise_scale_w", 0.8)
        self.length_scale  = tts_config.get("length_scale", 1.0)
        self.sdp_ratio     = tts_config.get("sdp_ratio", 0.2)
        self.split_interval= tts_config.get("split_interval", 1.0)
        self.line_split    = tts_config.get("line_split", False)

        self.emotion_service = EmotionService()
        
    def _load_default_model(self, tts_config: dict) -> None:
        """TTSModelHolder -> Default_Model 캐싱 + compile"""
        try:
            paths = TTSService._holder.model_files_dict[self.default_name]
        except KeyError:
            raise TTSError(
                "TTS_MODEL_NOT_FOUND",
                f"기본 TTS 모델 '{self.default_name}'을 찾을 수 없습니다."
            )
        
        TTSService._holder.get_model(self.default_name, str(paths[0]))
        model = TTSService._holder.current_model
        if model is None:
            raise TTSError(
                "TTS_MODEL_LOAD_ERROR",
                f"기본 TTS 모델 '{self.default_name}'을 로드할 수 없습니다."
            )
            
        if isinstance(model, torch.nn.Module):
            if torch.cuda.is_available() and torch.__version__ >= "2.0.0":
                model = torch.compile(
                    model,
                    mode="max-autotune",
                )
            model.eval()
            
        TTSService._model = model
    
    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        style: Optional[str] = None,
        style_weight: Optional[float] =  None
    ) -> dict:
        """
        주어진 텍스트를 TTS 모델을 사용하여 음성으로 합성하고,
        음성 파일의 URL과 스타일 정보를 반환합니다.
        
        Args:
            text (str): 합성할 텍스트
            
        Returns:
            dict: 음성 파일의 URL과 스타일 정보
        """
        if not text or not text.strip():
            raise TTSError("TTS_EMPTY_INPUT", "입력 텍스트가 비어 있습니다.")
        
        emotion_scores = self.emotion_service.predict(text)
        top_emotion = get_top_emotion(emotion_scores)
        
        if style is None or style_weight is None:
            style, style_weight = get_style_from_emotion(top_emotion)
        
        model = TTSService._model
        if model is None:
            raise TTSError("TTS_MODEL_NOT_LOADED", "TTS 모델이 로드되지 않았습니다.")
        
        try:
            segments: List[str] = [text]
            if self.line_split or len(text) > 200:
                segments = [
                    text[i : i + int(self.split_interval * 100)]
                    for i in range(0, len(text), int(self.split_interval * 100))
                ]
            
            audios = []
            for seg in segments:
                with torch.autocast(self.device, torch.float16, enabled=self.device == "cuda"):
                    sr, audio = model.infer(
                        text=seg,
                        language="JP",
                        reference_audio_path=None,
                        sdp_ratio=self.sdp_ratio,
                        noise=self.noise_scale,
                        noise_w=self.noise_scale_w,
                        length=self.length_scale,
                        line_split=self.line_split,
                        split_interval=self.split_interval,
                        assist_text=None,
                        assist_text_weight=0.0,
                        use_assist_text=False,
                        style=style,
                        style_weight=style_weight,
                        given_tone=None,
                        speaker_id=list(model.spk2id.values())[0],
                        pitch_scale=1.0,
                        intonation_scale=1.0,
                    )
                audios.append(audio)
                
            full_audio = torch.cat([torch.from_numpy(a) for a in audios]).numpy()
            
        except Exception as e:
            raise TTSError("TTS_INFERENCE_ERROR", f"TTS 합성 중 오류 발생: {str(e)}") from e

        filename = f"{uuid.uuid4()}.wav"
        outpath = self.outputs_dir / filename
        try:
            sf.write(str(outpath), full_audio, TTSService._sampling_rate)
        except Exception as e:
            raise TTSError("TTS_FILE_WRITE_ERROR", f"합성된 오디오 파일 저장 실패: {str(e)}") from e

        return {
            "audio_url": f"/outputs/{filename}",
            "emotion": top_emotion,
            "style": style,
            "style_weight": style_weight
        }