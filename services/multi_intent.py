# ====================================================================
#  File: services/multi_intent.py
# ====================================================================
"""
LunaMultiIntent 서비스 모듈

 - config/models.yaml 의 multi_intent 항목을 읽어와
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
from safetensors.torch import load_file
from transformers import AutoTokenizer, BitsAndBytesConfig
from utils.config import load_config_dict
from models.multiintent_model import MultiIntentClassifier


# ─────────────────────────────────────────────────────────────────────
# 예외 처리
# ─────────────────────────────────────────────────────────────────────
class MultiIntentError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code, self.detail = code, detail
        
# ─────────────────────────────────────────────────────────────────────
# MultiIntentService 클래스
# ─────────────────────────────────────────────────────────────────────
class MultiIntentService:
    """
    L.U.N.A. MultiIntent 서비스 클래스
    
    텍스트 -> 다중 인텐트 확률 분포 예측
    """
    
    _model: MultiIntentClassifier | None = None
    _tokenizer: AutoTokenizer | None = None
    
    def __init__(self) -> None:
        config = load_config_dict("models")["multi_intent"]
        self.threshold = config.get("threshold", 0.5)

        # 경로 설정
        self._model_dir = Path(config["model_dir"]).expanduser()
        self._tokenizer_dir = Path(config.get("tokenizer_dir", self._model_dir))
        self.label_list: List[str] = config["label_list"]
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if MultiIntentService._model is None or MultiIntentService._tokenizer is None:
            self._load_weights(config)

    @torch.inference_mode()
    def predict(self, text: str) -> Dict[str, float]:
        """
        주어진 텍스트에 대해 멀티 인텐트 예측을 수행하고,
        threshold 이상의 결과만 필터링해 반환합니다.

        Args:
            text (str): 입력 문장

        Returns:
            dict[str, float]: {intent_label: score} 형태의 딕셔너리
        """
        if not text.strip():
            raise MultiIntentError("INTENT_EMPTY_INPUT", "입력 텍스트가 비어 있습니다.")
        
        tok = MultiIntentService._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.autocast(self.device.type, torch.float16, enabled=self.device.type == "cuda"):
            out = MultiIntentService._model(**tok)
            logits = out["logits"].squeeze(0)
            probs  = torch.sigmoid(logits)

        return {
            self.id2label[i]: round(p.item(), 4)
            for i, p in enumerate(probs) if p.item() >= self.threshold
        }
        
    def _load_weights(self, config: dict) -> None:
        """모델 / 토크나이저 로드 + torch.compile 설정"""
        try:
            MultiIntentService._tokenizer = AutoTokenizer.from_pretrained(
                str(self._tokenizer_dir)
            )
            
            # 양자화 설정
            bnb_cfg = None
            if config.get("load_in_4bit"):
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                    )
                except Exception:
                    import logging
                    logging.warning("BitsAndBytesConfig not available. Using default precision.")
                    
            model_kwargs = {
                "model_name": config["model"]["name"],
                "num_labels": config["model"]["num_labels"],
            }
            
            if bnb_cfg is not None:
                model_kwargs["quant_cfg"] = bnb_cfg
                
            MultiIntentService._model = MultiIntentClassifier(**model_kwargs)
            
            sd = load_file(str(self._model_dir / "model.safetensors"), device="cpu")
            MultiIntentService._model.load_state_dict(sd, strict=False)
            MultiIntentService._model.to(self.device).eval()
            
            if torch.cuda.is_available() and torch.__version__ >= "2.0.0":
                MultiIntentService._model = torch.compile(
                    MultiIntentService._model,
                    mode="max-autotune",
                )
                
        except FileNotFoundError as e:
            raise MultiIntentError("INTENT_MODEL_NOT_FOUND", f"모델 파일 없음: {e}") from e
        except Exception as e:
            MultiIntentService._model = None
            raise MultiIntentError("INTENT_LOAD_FAIL", f"MultiIntent 초기화 실패: {e}") from e
