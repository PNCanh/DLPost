import torch
import torch.nn as nn


class ConditionalAttentionFusion(
    nn.Module
):

    def __init__(
        self,
        feature_dim,
        num_classes,
        num_explanations
    ):

        super().__init__()

        self.query = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.key = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.value = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.binary_head = nn.Linear(
            feature_dim,
            2
        )

        self.multiclass_head = nn.Linear(
            feature_dim,
            num_classes
        )

        self.explanation_head = nn.Linear(
            feature_dim,
            num_explanations
        )

    def forward(

        self,

        text_features,

        image_features
    ):

        q = self.query(
            text_features
        )

        k = self.key(
            image_features
        )

        v = self.value(
            image_features
        )

        scores = (

            q * k

        ).sum(

            dim=1,

            keepdim=True
        )

        weights = torch.sigmoid(
            scores
        )

        fused_features = (

            text_features

            +

            weights * v
        )

        return {

            "features":
                fused_features,

            "binary_logits":
                self.binary_head(
                    fused_features
                ),

            "multiclass_logits":
                self.multiclass_head(
                    fused_features
                ),

            "explanation_logits":
                self.explanation_head(
                    fused_features
                )
        }