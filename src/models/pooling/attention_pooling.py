"""
Attention Pooling (Nhóm 1 - 1 điểm)

Thay vì lấy CLS vector, học trọng số attention trên toàn bộ token hidden states.
- Tính attention score cho mỗi token qua một linear layer + tanh + linear
- Dùng softmax để chuẩn hóa thành weights
- Tính weighted sum của tất cả hidden states làm feature vector

So sánh:
- CLS pooling: features = hidden_states[:, 0]  (chỉ dùng token [CLS])
- Attention pooling: features = sum(alpha_i * hidden_states[:, i])  (dùng toàn bộ tokens)
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Attention Pooling thay thế CLS vector.
    
    Học trọng số attention (w) cố định cho toàn bộ token hidden states,
    sau đó tính weighted sum làm representation vector.
    
    Args:
        hidden_size: Kích thước hidden state từ encoder (vd: 768 cho BERT-base)
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Attention network: hidden_size -> hidden_size -> 1
        # Tính score cho mỗi token
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - Toàn bộ hidden states
            attention_mask: (batch_size, seq_len) - Mask cho padding tokens
            
        Returns:
            pooled: (batch_size, hidden_size) - Attention-pooled vector
        """
        # Tính attention scores: (batch_size, seq_len, 1)
        attn_scores = self.attention(hidden_states)

        # Mask padding tokens (đặt score = -inf để softmax → 0)
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            mask = attention_mask.unsqueeze(-1).float()
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax trên chiều seq_len: (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum: (batch_size, seq_len, hidden_size) * (batch_size, seq_len, 1)
        # -> sum trên dim=1 -> (batch_size, hidden_size)
        pooled = (hidden_states * attn_weights).sum(dim=1)

        return pooled
