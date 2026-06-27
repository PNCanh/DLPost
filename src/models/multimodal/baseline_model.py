import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as models

from models.layers.residual_mlp import ResidualMLP
from models.layers.task_attention import TaskAttention

class BaselineMultimodalModel(nn.Module):
    """
    Baseline Model:
    - Text: PhoBERT
    - Image: ResNet50
    - Fusion: Simple Concatenation
    - Heads: Parallel Multi-task Learning (TaskAttention + ResidualMLP)
    """
    def __init__(self, num_binary_classes=1, num_multi_classes=8, num_explain_classes=10):
        super(BaselineMultimodalModel, self).__init__()
        
        # 1. Text Backbone (PhoBERT)
        self.text_model = AutoModel.from_pretrained("vinai/phobert-base")
        text_hidden_size = self.text_model.config.hidden_size # 768
        
        # 2. Image Backbone (ResNet)
        resnet = models.resnet50(pretrained=True)
        # Bỏ layer classification cuối cùng
        self.image_model = nn.Sequential(*list(resnet.children())[:-1])
        image_hidden_size = 2048 # ResNet50 output dim
        
        # 3. Fusion Layer
        self.fusion_dim = text_hidden_size + image_hidden_size
        
        # 4. Multi-task Heads
        self.task_attention = TaskAttention(hidden_dim=self.fusion_dim, num_tasks=3)
        
        # Task 1: Binary Classification (Scam vs Legitimate)
        self.head_binary = ResidualMLP(
            input_dim=self.fusion_dim, 
            hidden_dim=self.fusion_dim // 2, 
            output_dim=num_binary_classes
        )
        
        # Task 2: Multi-class Classification (Job scam, Investment scam, v.v.)
        self.head_category = ResidualMLP(
            input_dim=self.fusion_dim,
            hidden_dim=self.fusion_dim // 2,
            output_dim=num_multi_classes
        )
        
        # Task 3: Explanation Classification (Multi-label)
        self.head_explanation = ResidualMLP(
            input_dim=self.fusion_dim,
            hidden_dim=self.fusion_dim // 2,
            output_dim=num_explain_classes
        )

    def forward(self, input_ids, attention_mask, image):
        # Lấy đặc trưng văn bản
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # Sử dụng pooler_output hoặc token [CLS]
        text_features = text_outputs.last_hidden_state[:, 0, :] # [batch, 768]
        
        # Lấy đặc trưng hình ảnh
        image_features = self.image_model(image) # [batch, 2048, 1, 1]
        image_features = image_features.view(image_features.size(0), -1) # [batch, 2048]
        
        # Fusion (Concatenation)
        fused_features = torch.cat([text_features, image_features], dim=1) # [batch, 2816]
        
        # Task Attention (tách feature cho 3 task)
        task_features = self.task_attention(fused_features)
        
        # Phân loại cho từng task
        out_binary = self.head_binary(task_features[0])
        out_category = self.head_category(task_features[1])
        out_explanation = self.head_explanation(task_features[2])
        
        return out_binary, out_category, out_explanation
