# ====================================================================
#  File: services/emotion.py
# ====================================================================
"""
LunaEmotion 서비스 모듈

 - config/models.yaml 의 emotion 항목을 읽어와
    로컬 체크포인트 또는 허브 모델을 로드합니다.
 - predict(text: str) 호출 시,
    threshold 이상의 레이블과 점수를 dict로 반환합니다.
"""

from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from utils.config import load_config_dict
from models.emotion_model import EmotionClassifier, compute_pos_weight
from datasets import load_from_disk

class EmotionService:
    def __init__(self):
        config = load_config_dict("models")["emotion"]
        self.threshold = config.get('threshold', 0.5)

        model_dir = Path(config["model_dir"]).expanduser()
        tokenizer_dir = Path(config.get("tokenizer_dir", model_dir))

        self.label_list = config["label_list"]
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = config["model"]["name"]
        num_labels = config["model"]["num_labels"]
        dropout_prob = config["model"].get("dropout_prob", 0.1)

        self.model = EmotionClassifier(
            model_name=model_name,
            num_labels=num_labels,
            dropout_prob=dropout_prob,
            pos_weight=torch.ones(len(self.label_list))
        )
        state_dict = load_file(str(model_dir / "model.safetensors"), device=self.device.type)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        
    def predict(self, text: str) -> dict[str, float]:
        """
        주어진 텍스트에 대해 감정 예측을 수행하고,
        threshold 이상의 결과만 필터링해 반환합니다.

        Args:
            text (str): 입력 문장

        Returns:
            dict[str, float]: {label: score} 형태의 딕셔너리
        """
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        enc = {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device)
        }
        
        with torch.no_grad():
            logits = self.model(**enc).logits.squeeze(0)
            probs = torch.sigmoid(logits)

        return {
            self.id2label[i]: round(probs[i].item(), 4)
            for i in range(probs.size(0))
            if probs[i].item() >= self.threshold
        }