import torch
import torch.nn as nn


class GatedFusion(
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

        self.keyword_dim = keyword_dim

        self.gate = nn.Sequential(

            nn.Linear(
                feature_dim * 2 + keyword_dim,
                feature_dim
            ),

            nn.Sigmoid()
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
        
        combined_list = [text_features, image_features]
        if keyword_vector is not None:
            combined_list.append(keyword_vector)

        combined = torch.cat(
            combined_list,
            dim=1
        )

        gate = self.gate(
            combined
        )

        fused_features = (

            gate
            * text_features

            +

            (
                1 - gate
            )
            * image_features
        )
        
        if keyword_vector is not None:
            final_features = torch.cat([fused_features, keyword_vector], dim=1)
        else:
            final_features = fused_features

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