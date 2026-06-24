import torch
import torch.nn as nn

from transformers import (
    ViTModel
)


class ViTClassifier(
    nn.Module
):

    def __init__(
        self,
        num_multiclass,
        num_explanations,
        model_name=
        "google/vit-base-patch16-224",
        dropout=0.3
    ):

        super().__init__()

        self.backbone = (
            ViTModel
            .from_pretrained(
                model_name
            )
        )

        feature_dim = (
            self.backbone.config.hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.binary_head = nn.Linear(

            feature_dim,

            2
        )

        self.multiclass_head = nn.Linear(

            feature_dim,

            num_multiclass
        )

        self.explanation_head = nn.Linear(

            feature_dim,

            num_explanations
        )

    def forward(
        self,
        pixel_values
    ):

        outputs = (
            self.backbone(
                pixel_values=
                pixel_values
            )
        )

        image_features = (
            outputs
            .last_hidden_state[
                :,
                0
            ]
        )

        image_features = (
            self.dropout(
                image_features
            )
        )

        binary_logits = (
            self.binary_head(
                image_features
            )
        )

        multiclass_logits = (
            self.multiclass_head(
                image_features
            )
        )

        explanation_logits = (
            self.explanation_head(
                image_features
            )
        )

        return {

            "features":
                image_features,

            "binary_logits":
                binary_logits,

            "multiclass_logits":
                multiclass_logits,

            "explanation_logits":
                explanation_logits
        }