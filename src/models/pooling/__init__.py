"""
Pooling Strategies Module

Các chiến lược lấy vector đặc trưng từ hidden states của text encoder:
- cls: Dùng CLS token (mặc định)
- attention_pooling: Attention-weighted sum toàn bộ token hidden states (không dùng CLS)
- gated_cls: Kết hợp CLS + attention pooling qua learned gate
"""

from models.pooling.attention_pooling import AttentionPooling
from models.pooling.gated_pooling import GatedCLSPooling
