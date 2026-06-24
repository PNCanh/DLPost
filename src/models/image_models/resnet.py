import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


class ResNetClassifier(
    nn.Module
):

    def __init__(
        self,
        num_multiclass,
        num_explanations,
        pretrained=True,
        dropout=0.3
    ):

        super().__init__()

        if pretrained:

            self.backbone = resnet50(

                weights=
                ResNet50_Weights.DEFAULT
            )

        else:

            self.backbone = resnet50(
                weights=None
            )

        feature_dim = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

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
        images
    ):

        image_features = (
            self.backbone(images)
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