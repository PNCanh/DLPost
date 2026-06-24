"""
Conditional Attention Fusion (Nhóm 1 - 2 điểm)

Vector (w) dùng tính trọng số attention KHÔNG cố định, mà phụ thuộc vào aspect (keyword_vector).

Ý tưởng:
- Trong attention fusion thông thường, vector w là learned parameter cố định
- Ở đây, w được tạo ra từ keyword_vector (aspect) qua một linear projection
- Khi aspect khác nhau → w khác → attention weights khác → fused features khác

Formula:
    w = W_aspect * keyword_vector + b_aspect   (aspect-dependent query)
    q = W_q * text_features
    k = W_k * image_features  
    v = W_v * image_features
    scores = (q + w) * k  (aspect-conditioned attention)
    weights = sigmoid(scores)
    fused = text_features + weights * v
"""

import torch
import torch.nn as nn


class ConditionalAttentionFusion(
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

        # Aspect-dependent projection: keyword_vector → attention query modifier
        # Khi keyword_dim > 0, vector w phụ thuộc vào aspect
        if keyword_dim > 0:
            self.aspect_projection = nn.Sequential(
                nn.Linear(keyword_dim, feature_dim),
                nn.Tanh()
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

        q = self.query(
            text_features
        )

        k = self.key(
            image_features
        )

        v = self.value(
            image_features
        )

        # Conditional: nếu có keyword_vector, điều chỉnh query theo aspect
        if keyword_vector is not None and self.keyword_dim > 0:
            # aspect_modifier: (batch_size, feature_dim)
            aspect_modifier = self.aspect_projection(keyword_vector)
            # Cộng aspect modifier vào query → attention phụ thuộc aspect
            q = q + aspect_modifier

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

        # Concat keyword_vector vào feature trước classification heads
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