import torch
import torch.nn as nn


class AttentionFusion(
    nn.Module
):

    def __init__(
        self,
        feature_dim,
        num_classes,
        num_explanations,
        keyword_dim=0
    ):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                feature_dim * 2,
                feature_dim
            ),

            nn.Tanh(),

            nn.Linear(
                feature_dim,
                2
            )
        )

        self.binary_head = nn.Linear(
            feature_dim + keyword_dim,
            2
        )

        self.multiclass_head = nn.Linear(
            feature_dim + keyword_dim,
            num_classes
        )

        self.explanation_head = nn.Linear(
            feature_dim + keyword_dim,
            num_explanations
        )

    def forward(

        self,

        text_features,

        image_features,

        keyword_vector=None
    ):

        combined = torch.cat(

            [
                text_features,
                image_features
            ],

            dim=1
        )

        weights = torch.softmax(

            self.attention(
                combined
            ),

            dim=1
        )

        text_weight = weights[
            :,
            0
        ].unsqueeze(1)

        image_weight = weights[
            :,
            1
        ].unsqueeze(1)

        fused_features = (

            text_weight
            * text_features

            +

            image_weight
            * image_features
        )

        final_features = fused_features
        if keyword_vector is not None:
            final_features = torch.cat([final_features, keyword_vector], dim=1)

        return {

            "features":
                fused_features,

            "binary_logits":
                self.binary_head(
                    final_features
                ),

            "multiclass_logits":
                self.multiclass_head(
                    final_features
                ),

            "explanation_logits":
                self.explanation_head(
                    final_features
                )
        }