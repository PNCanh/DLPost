"""
Gated CLS + Attention Pooling (Nhóm 1 - 1 điểm)

Kết hợp CLS vector và attention pooling vector bằng một cổng học được (learned gate).

Formula:
    gate = sigmoid(W_g * [cls; attn_pool] + b_g)
    output = gate * cls + (1 - gate) * attn_pool

Ý tưởng:
- CLS token tốt cho việc nắm bắt ngữ nghĩa tổng thể
- Attention pooling tốt cho việc tập trung vào các token quan trọng
- Gate tự học cách kết hợp tối ưu giữa hai nguồn thông tin
"""

import torch
import torch.nn as nn

from models.pooling.attention_pooling import AttentionPooling


class GatedCLSPooling(nn.Module):
    """
    Gated fusion giữa CLS vector và Attention Pooling vector.
    
    Args:
        hidden_size: Kích thước hidden state từ encoder (vd: 768)
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Attention pooling component
        self.attention_pooling = AttentionPooling(hidden_size)

        # Gate network: nhận concat [cls, attn_pool] → sigmoid → gate value
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            fused: (batch_size, hidden_size) - Gated combination of CLS and attention pooling
        """
        # CLS vector: lấy token đầu tiên
        cls_vector = hidden_states[:, 0]

        # Attention pooling vector
        attn_pool_vector = self.attention_pooling(hidden_states, attention_mask)

        # Concat CLS và attention pooling → tính gate
        combined = torch.cat([cls_vector, attn_pool_vector], dim=1)
        gate_value = self.gate(combined)

        # Gated fusion: g * CLS + (1-g) * attn_pool
        fused = gate_value * cls_vector + (1 - gate_value) * attn_pool_vector

        return fused
