# ====================================================================
#  File: services/emotion.py
# ====================================================================
"""
LunaEmotion 서비스 모듈

 - config/models.yaml 의 emotion 항목을 읽어와
    로컬 체크포인트 또는 허브 모델을 로드합니다.
 - predict(text: str) 호출 시,
    threshold 이상의 레이블과 점수를 dict로 반환합니다.
    
    2025/07/13
     - 예외처리 추가 및 전체 코드 옵티마이징
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from safetensors.torch import load_file
from utils.config import load_config_dict
from models.emotion_model import EmotionClassifier, compute_pos_weight


# ─────────────────────────────────────────────────────────────────────
# 예외 처리
# ─────────────────────────────────────────────────────────────────────
class EmotionError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code, self.detail = code, detail
        
# ─────────────────────────────────────────────────────────────────────
# EmotionService 클래스
# ─────────────────────────────────────────────────────────────────────
class EmotionService:
    """
    L.U.N.A. Emotion 서비스 클래스
    
    텍스트 -> 다중 감정 확률 분포 예측
    """

    _model: EmotionClassifier | None = None
    _tokenizer: AutoTokenizer | None = None
    
    def __init__(self):
        config = load_config_dict("models")["emotion"]
        self.threshold = config.get('threshold', 0.5)

        # 경로 설정
        self._model_dir = Path(config["model_dir"]).expanduser()
        self._tokenizer_dir = Path(config.get("tokenizer_dir", self._model_dir))
        self.label_list: List[str] = config["label_list"]
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if EmotionService._model is None or EmotionService._tokenizer is None:
            self._load_weights(config)
            
    # ─────────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict(self, text: str) -> Dict[str, float]:
        """
        주어진 텍스트에 대해 감정 예측을 수행하고,
        threshold 이상의 결과만 필터링해 반환합니다.

        Args:
            text (str): 입력 문장

        Returns:
            dict[str, float]: {label: score} 형태의 딕셔너리
        """
        if not text.strip():
            raise EmotionError("EMO_EMPTY_INPUT", "입력 텍스트가 비어 있습니다.")

        tok = EmotionService._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.autocast(self.device.type, torch.float16, enabled=self.device.type == "cuda"):
            logits = EmotionService._model(**tok).logits.squeeze(0)
            probs = torch.sigmoid(logits)

        return {
            self.id2label[i]: round(p.item(), 4)
            for i, p in enumerate(probs) if p.item() >= self.threshold
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # internal helper
    # ─────────────────────────────────────────────────────────────────────
    def _load_weights(self, config: dict) -> None:
        """모델 / 토크나이저 로드 및 캐싱"""
        try:
            EmotionService._tokenizer = AutoTokenizer.from_pretrained(
                str(self._tokenizer_dir)
            )
            
            bnb_cfg = None
            if config.get("load_in_4bit"):
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4"
                )
                
            EmotionService._model = EmotionClassifier(
                model_name = config["model"]["name"],
                num_labels = config["model"]["num_labels"],
                dropout_prob = config["model"].get("dropout_prob", 0.1),
                pos_weight   = torch.ones(len(self.label_list)),
                # quant_cfg    = bnb_cfg, #TODO 4bit quantization
            )
            
            sd_path = self._model_dir / "model.safetensors"
            state = load_file(str(sd_path), device="cpu")
            EmotionService._model.load_state_dict(state, strict=False)
            EmotionService._model.to(self.device).eval()
            
            if torch.cuda.is_available() and torch.__version__ >= "2.0.0":
                EmotionService.model = torch.compile(
                    EmotionService._model, mode="reduce-overhead"
                )
        
        except FileNotFoundError as e:
            raise EmotionError("EMO_MODEL_NOT_FOUND", f"모델 파일 없음: {e}") from e
        except Exception as e:
            # 어떤 이유로든 초기화 실패 시 추후 predict 차단
            EmotionService._model = None
            raise EmotionError("EMO_LOAD_FAIL", f"Emotion 모델 초기화 실패: {e}") from e