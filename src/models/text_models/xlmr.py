"""
XLM-RoBERTa Classifier

Text classifier sử dụng XLM-RoBERTa (xlm-roberta-base) làm backbone.
Hỗ trợ 3 chiến lược pooling:
- cls: Dùng CLS token (mặc định)
- attention_pooling: Attention-weighted sum toàn bộ hidden states
- gated_cls: Kết hợp CLS + attention pooling qua learned gate
"""

import torch
import torch.nn as nn

from transformers import (
    AutoModel
)


class XLMRClassifier(
    nn.Module
):

    def __init__(
        self,
        model_name,
        num_multiclass,
        num_explanations,
        dropout=0.1,
        pooling_strategy="cls"
    ):

        super().__init__()

        self.pooling_strategy = pooling_strategy

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Khởi tạo pooling module nếu cần
        if pooling_strategy == "attention_pooling":
            from models.pooling.attention_pooling import AttentionPooling
            self.pooling = AttentionPooling(hidden_size)
        elif pooling_strategy == "gated_cls":
            from models.pooling.gated_pooling import GatedCLSPooling
            self.pooling = GatedCLSPooling(hidden_size)

        self.binary_head = nn.Linear(
            hidden_size,
            2
        )

        self.multiclass_head = nn.Linear(
            hidden_size,
            num_multiclass
        )

        self.explanation_head = nn.Linear(
            hidden_size,
            num_explanations
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        # Chọn pooling strategy
        if self.pooling_strategy == "cls":
            features = (
                outputs.last_hidden_state[
                    :, 0
                ]
            )
        elif self.pooling_strategy in ("attention_pooling", "gated_cls"):
            features = self.pooling(
                outputs.last_hidden_state,
                attention_mask
            )

        features = self.dropout(
            features
        )

        binary_logits = (
            self.binary_head(
                features
            )
        )

        multiclass_logits = (
            self.multiclass_head(
                features
            )
        )

        explanation_logits = (
            self.explanation_head(
                features
            )
        )

        return {

            "features":
                features,

            "binary_logits":
                binary_logits,

            "multiclass_logits":
                multiclass_logits,

            "explanation_logits":
                explanation_logits
        }