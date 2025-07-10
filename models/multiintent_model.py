# ====================================================================
#  File: models/multiintent_model.py
# ====================================================================
"""
LunaMultiIntent 모델
"""

import torch
import torch.nn as nn
from transformers import AutoModel

# ----------
# Core Model Definition
# ----------

class MultiIntentClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> dict:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls_emb = encoder_outputs.last_hidden_state[:, 0]
        
        logits = self.classifier(cls_emb)

        output = {"logits": logits}

        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels.float())

        return output
