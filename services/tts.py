# ====================================================================
#  File: services/tts.py
# ====================================================================
"""
L.U.N.A. TTS 합성 서비스 모듈

• luna_core.tts_model.TTSModelHolder를 사용하여 실제 음성 합성을 실행
• 합성된 오디오는 outputs 디렉토리에 WAV로 저장되고, 그 URL을 반환
"""

import os
import uuid

import soundfile as sf
from pathlib import Path

from fastapi import HTTPException

from utils.config import load_config_dict
from utils.style_map import get_style_from_emotion, get_top_emotion
from services.emotion import EmotionService
from models.tts_model import TTSModelHolder

class TTSService:
    def __init__(self, device: str):
        tts_config = load_config_dict("models")["tts"]
        
        model_dir = Path(tts_config["model_dir"]).expanduser()
        self.device = device
        
        self.model_holder = TTSModelHolder(str(model_dir), device)
        
        self.default_model = tts_config.get("default_model")
        if self.default_model not in self.model_holder.model_names:
            raise RuntimeError(f"[L.U.N.A.] Default TTS model '{self.default_model}' not found in {model_dir}")
        self.default_model_paths = self.model_holder.model_files_dict[self.default_model]

        self.sampling_rate = tts_config.get("sampling_rate", 44100)
        self.noise_scale = tts_config.get("noise_scale", 0.6)
        self.noise_scale_w = tts_config.get("noise_scale_w", 0.8)
        self.length_scale = tts_config.get("length_scale", 1.0)
        self.sdp_ratio = tts_config.get("sdp_ratio", 0.2)
        self.split_interval = tts_config.get("split_interval", 0.5)
        self.line_split = tts_config.get("line_split", False)
        
        self.emotion_service = EmotionService()
    
    def synthesize(self, text: str, style: str = "Neutral", style_weight: float = 1.0) -> str:
        """
        주어진 텍스트를 TTS 모델을 사용하여 음성으로 합성하고,
        합성된 오디오 파일의 URL을 반환합니다.
        
        Args:
            text (str): 합성할 텍스트
            
        Returns:
            str: 합성된 오디오 파일의 URL
        """
        model_path = str(self.default_model_paths[0])
        self.model_holder.get_model(self.default_model, model_path)
        model = self.model_holder.current_model
        if model is None:
            raise HTTPException(status_code=500, detail="TTS model not loaded")
        
        emotion_scores = self.emotion_service.predict(text)
        top_emotion = get_top_emotion(emotion_scores)
        style, style_weight = get_style_from_emotion(top_emotion)
        
        try:
            sr, audio = model.infer(
                text=text,
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
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS inference error: {e}")

        filename = f"{uuid.uuid4()}.wav"
        filepath = os.path.join("outputs", filename)
        sf.write(filepath, audio, sr)

        return f"/outputs/{filename}"