"""
Model Factory

Factory để khởi tạo model multimodal dựa trên cấu hình từ model_config.py.

Hỗ trợ:
- Text models: phobert, vibert, xlmr (với pooling_strategy: cls, attention_pooling, gated_cls)
- Image models: resnet, vit
- Fusion strategies: concat, attention, gated, conditional_attention
"""

import torch
import torch.nn as nn

# Import Text Models
from models.text_models.phobert import PhoBERTClassifier
from models.text_models.vibert import ViBERTClassifier
from models.text_models.xlmr import XLMRClassifier

# Import Image Models
from models.image_models.resnet import ResNetClassifier
from models.image_models.vit import ViTClassifier

# Import Fusion Models
from models.fusion.attention import AttentionFusion
from models.fusion.conditional_attention import ConditionalAttentionFusion
from models.fusion.gated import GatedFusion


class MultimodalModel(nn.Module):
    """
    Mô hình kết hợp (Multimodal) tự động khởi tạo dựa trên model_config.
    
    Hỗ trợ cấu hình linh hoạt:
    - pooling_strategy: Cách lấy vector đặc trưng từ text encoder
    - fusion_strategy: Cách kết hợp text + image features
    """
    def __init__(self, config, num_multiclass=8, num_explanations=10):
        super().__init__()
        
        # Lấy pooling strategy từ config
        pooling_strategy = config.get("pooling_strategy", "cls")
        
        # 1. Text Model
        text_model_name = config.get("text_model", "phobert")
        if text_model_name == "phobert":
            self.text_model = PhoBERTClassifier(
                "vinai/phobert-base", num_multiclass, num_explanations,
                pooling_strategy=pooling_strategy
            )
        elif text_model_name == "vibert":
            self.text_model = ViBERTClassifier(
                "FPTAI/vibert-base-cased", num_multiclass, num_explanations,
                pooling_strategy=pooling_strategy
            )
        elif text_model_name == "xlmr":
            self.text_model = XLMRClassifier(
                "xlm-roberta-base", num_multiclass, num_explanations,
                pooling_strategy=pooling_strategy
            )
        else:
            raise ValueError(f"Unknown text model: {text_model_name}")
            
        # 2. Image Model
        image_model_name = config.get("image_model", "resnet")
        if image_model_name == "resnet":
            self.image_model = ResNetClassifier(num_multiclass=num_multiclass, num_explanations=num_explanations)
        elif image_model_name == "vit":
            self.image_model = ViTClassifier(num_multiclass=num_multiclass, num_explanations=num_explanations, model_name="google/vit-base-patch16-224")
        else:
            raise ValueError(f"Unknown image model: {image_model_name}")

        # Lấy feature_dim (thường là 768 cho BERT/ViT, 2048 cho ResNet50)
        self.feature_dim = 768 
        
        # 3. Fusion Strategy
        fusion_strategy = config.get("fusion_strategy", "concat")
        self.fusion_strategy = fusion_strategy
        self.keyword_dim = config.get("keyword_dim", 44)  # Dựa trên cấu trúc scam_keywords.json
        
        if fusion_strategy == "attention":
            self.fusion = AttentionFusion(self.feature_dim, num_multiclass, num_explanations, keyword_dim=self.keyword_dim)
        elif fusion_strategy == "conditional_attention":
            self.fusion = ConditionalAttentionFusion(self.feature_dim, num_multiclass, num_explanations, keyword_dim=self.keyword_dim)
        elif fusion_strategy == "gated":
            self.fusion = GatedFusion(self.feature_dim, num_multiclass, num_explanations, keyword_dim=self.keyword_dim)
        elif fusion_strategy == "concat":
            # Concat đơn giản
            self.binary_head = nn.Linear(self.feature_dim * 2 + self.keyword_dim, 2)
            self.multiclass_head = nn.Linear(self.feature_dim * 2 + self.keyword_dim, num_multiclass)
            self.explanation_head = nn.Linear(self.feature_dim * 2 + self.keyword_dim, num_explanations)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

    def forward(self, input_ids, attention_mask, images, keyword_vector=None):
        # Trích xuất đặc trưng văn bản
        text_outputs = self.text_model(input_ids, attention_mask)
        text_features = text_outputs["features"]
        
        # Trích xuất đặc trưng hình ảnh
        image_outputs = self.image_model(images)
        image_features = image_outputs["features"]
        
        # Projection cho ResNet nếu cần (ResNet thường 2048, cần project về 768 để fusion khớp)
        if image_features.shape[1] != text_features.shape[1]:
            if not hasattr(self, 'img_proj'):
                self.img_proj = nn.Linear(image_features.shape[1], text_features.shape[1]).to(image_features.device)
            image_features = self.img_proj(image_features)

        # Kết hợp đặc trưng
        if self.fusion_strategy == "concat":
            fused_features = torch.cat([text_features, image_features], dim=1)
            final_features = fused_features
            if keyword_vector is not None:
                final_features = torch.cat([fused_features, keyword_vector], dim=1)

            binary_logits = self.binary_head(final_features) + text_outputs["binary_logits"] + image_outputs["binary_logits"]
            multiclass_logits = self.multiclass_head(final_features) + text_outputs["multiclass_logits"] + image_outputs["multiclass_logits"]
            explanation_logits = self.explanation_head(final_features) + text_outputs["explanation_logits"] + image_outputs["explanation_logits"]
            
            return {
                "features": fused_features,
                "binary_logits": binary_logits,
                "multiclass_logits": multiclass_logits,
                "explanation_logits": explanation_logits
            }
        else:
            fusion_outputs = self.fusion(text_features, image_features, keyword_vector)
            fusion_outputs["binary_logits"] = fusion_outputs["binary_logits"] + text_outputs["binary_logits"] + image_outputs["binary_logits"]
            fusion_outputs["multiclass_logits"] = fusion_outputs["multiclass_logits"] + text_outputs["multiclass_logits"] + image_outputs["multiclass_logits"]
            fusion_outputs["explanation_logits"] = fusion_outputs["explanation_logits"] + text_outputs["explanation_logits"] + image_outputs["explanation_logits"]
            return fusion_outputs


class ModelFactory:
    """
    Factory để khởi tạo model và loss dựa trên cấu hình.
    """
    @staticmethod
    def build_model(config, num_multiclass=8, num_explanations=10):
        return MultimodalModel(config, num_multiclass, num_explanations)
