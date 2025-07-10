# ====================================================================
#  File: models/emotion_model.py
# ====================================================================
"""
LunaEmotion 모델
"""

import torch, os
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

# ----------
# Helper functions
# ----------

def compute_pos_weight(label_tensor: torch.Tensor) -> torch.Tensor:
    """
    BCEWithLogitsLoss의 pos_weight를 계산합니다.
    """
    
    pos_counts = label_tensor.sum(dim=0)
    neg_counts = label_tensor.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8)
    return pos_weight

# ----------
# Core Model Definition
# ----------

class EmotionClassifier(torch.nn.Module):
    """
    Pretrained Transformer 기반 다중 레이블 감정 분류 모델
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout_prob: float = 0.1,
        pos_weight: torch.Tensor = None
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=False
        )

        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )

        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(self.config.hidden_size, num_labels)

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        token_type_ids: torch.Tensor = None
    ) -> SequenceClassifierOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        cls_emb = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_emb)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )