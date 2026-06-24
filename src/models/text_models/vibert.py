import torch
import torch.nn as nn

from transformers import (
    AutoModel
)


class ViBERTClassifier(
    nn.Module
):

    def __init__(
        self,
        model_name,
        num_multiclass,
        num_explanations,
        dropout=0.1
    ):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

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

        features = (
            outputs.last_hidden_state[
                :, 0
            ]
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