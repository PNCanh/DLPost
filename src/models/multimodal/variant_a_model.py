import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel

from models.layers.residual_mlp import ResidualMLP
from models.layers.task_attention import TaskAttention
from models.layers.weighted_layer_sum import WeightedLayerSum
from models.layers.multi_sample_dropout import MultiSampleDropout

class VariantAModel(nn.Module):
    """
    Variant A Model:
    - Text: XLMR (với Weighted Layer Sum)
    - Image: ViT
    - Fusion: Cross-modal Attention / Concatenation
    - Heads: Multi-sample Dropout + Residual MLP + Task Attention
    """
    def __init__(self, num_binary_classes=1, num_multi_classes=8, num_explain_classes=10):
        super(VariantAModel, self).__init__()
        
        # 1. Text Backbone (XLMR)
        self.text_model = AutoModel.from_pretrained("xlm-roberta-base", output_hidden_states=True)
        self.weighted_layer_sum = WeightedLayerSum(num_layers=12)
        text_hidden_size = self.text_model.config.hidden_size
        
        # 2. Image Backbone (ViT)
        self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        image_hidden_size = self.image_model.config.hidden_size
        
        # 3. Fusion Layer
        self.fusion_dim = text_hidden_size + image_hidden_size
        
        # 4. Multi-task Heads
        self.task_attention = TaskAttention(hidden_dim=self.fusion_dim, num_tasks=3)
        self.multi_dropout = MultiSampleDropout(p=0.1, n_samples=5)
        
        self.head_binary = ResidualMLP(self.fusion_dim, self.fusion_dim // 2, num_binary_classes)
        self.head_category = ResidualMLP(self.fusion_dim, self.fusion_dim // 2, num_multi_classes)
        self.head_explanation = ResidualMLP(self.fusion_dim, self.fusion_dim // 2, num_explain_classes)

    def forward(self, input_ids, attention_mask, image):
        # Text
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # Sử dụng Weighted Layer Sum
        text_weighted = self.weighted_layer_sum(text_outputs.hidden_states)
        text_features = text_weighted[:, 0, :] # Lấy token [CLS]
        
        # Image
        image_outputs = self.image_model(pixel_values=image)
        image_features = image_outputs.last_hidden_state[:, 0, :] # Lấy token [CLS]
        
        # Fusion
        fused_features = torch.cat([text_features, image_features], dim=1)
        
        # Task Attention
        task_features = self.task_attention(fused_features)
        
        # Phân loại với Multi-sample Dropout
        out_binary = self.multi_dropout(task_features[0], self.head_binary)
        out_category = self.multi_dropout(task_features[1], self.head_category)
        out_explanation = self.multi_dropout(task_features[2], self.head_explanation)
        
        return out_binary, out_category, out_explanation
