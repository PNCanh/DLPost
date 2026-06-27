import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as models

from models.layers.residual_mlp import ResidualMLP
from models.layers.task_attention import TaskAttention

class VariantBModel(nn.Module):
    """
    Variant B Model:
    - Text: ViBERT
    - Image: ResNet50
    - Fusion: Gated Multimodal Fusion
    - Heads: Parallel Multi-task Learning
    """
    def __init__(self, num_binary_classes=1, num_multi_classes=8, num_explain_classes=10):
        super(VariantBModel, self).__init__()
        
        # 1. Text (ViBERT)
        self.text_model = AutoModel.from_pretrained("FPTAI/vibert-base-cased")
        text_hidden_size = self.text_model.config.hidden_size
        
        # 2. Image (ResNet)
        resnet = models.resnet50(pretrained=True)
        self.image_model = nn.Sequential(*list(resnet.children())[:-1])
        image_hidden_size = 2048
        
        # 3. Gated Fusion
        self.fusion_dim = text_hidden_size # Project image về chung không gian với text
        self.image_proj = nn.Linear(image_hidden_size, self.fusion_dim)
        self.gate = nn.Linear(self.fusion_dim * 2, self.fusion_dim)
        
        # 4. Multi-task Heads
        self.task_attention = TaskAttention(hidden_dim=self.fusion_dim, num_tasks=3)
        self.head_binary = ResidualMLP(self.fusion_dim, self.fusion_dim // 2, num_binary_classes)
        self.head_category = ResidualMLP(self.fusion_dim, self.fusion_dim // 2, num_multi_classes)
        self.head_explanation = ResidualMLP(self.fusion_dim, self.fusion_dim // 2, num_explain_classes)

    def forward(self, input_ids, attention_mask, image):
        # Text
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] 
        
        # Image
        image_features = self.image_model(image)
        image_features = image_features.view(image_features.size(0), -1) 
        image_features = self.image_proj(image_features)
        
        # Gated Fusion
        concat = torch.cat([text_features, image_features], dim=1)
        z = torch.sigmoid(self.gate(concat))
        fused_features = z * text_features + (1 - z) * image_features
        
        # Heads
        task_features = self.task_attention(fused_features)
        out_binary = self.head_binary(task_features[0])
        out_category = self.head_category(task_features[1])
        out_explanation = self.head_explanation(task_features[2])
        
        return out_binary, out_category, out_explanation
