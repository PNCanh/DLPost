"""
Focal Loss (Nhóm 2 - 2 điểm)

Tăng trọng số cho mẫu khó (hard examples), giảm ảnh hưởng của mẫu dễ (easy examples).

Formula:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
Trong đó:
- p_t: xác suất dự đoán đúng của model
- gamma (focusing parameter): gamma > 0 → giảm loss cho mẫu dễ
  + gamma = 0: Focal Loss = CrossEntropy thông thường
  + gamma = 2 (default): Mẫu dễ (p_t > 0.9) gần như không đóng góp loss
- alpha: hệ số cân bằng

Hai phiên bản:
- FocalLoss: Cho multiclass classification (dựa trên CrossEntropy)
- BinaryFocalLoss: Cho binary classification (dựa trên BCEWithLogits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss cho Multiclass Classification.
    
    Dựa trên CrossEntropy, thêm modulating factor (1 - p_t)^gamma
    để focus vào hard examples.
    
    Args:
        alpha: Hệ số cân bằng (mặc định: 1.0)
        gamma: Focusing parameter (mặc định: 2.0, giá trị phổ biến: 0.5, 1, 2, 5)
        reduction: 'mean', 'sum', hoặc 'none'
    """

    def __init__(
        self,
        alpha=1.0,
        gamma=2.0,
        reduction="mean"
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits,
        targets
    ):
        """
        Args:
            logits: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class indices
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        # p_t = probability of correct class
        pt = torch.exp(-ce_loss)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_loss = (
            self.alpha
            * ((1 - pt) ** self.gamma)
            * ce_loss
        )

        if self.reduction == "mean":
            return focal_loss.mean()

        if self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss cho Binary Classification (sử dụng BCEWithLogits).
    
    Tương tự FocalLoss nhưng cho bài toán binary/multi-label.
    
    Args:
        alpha: Hệ số cân bằng (mặc định: 0.25 - common cho imbalanced data)
        gamma: Focusing parameter (mặc định: 2.0)
        reduction: 'mean', 'sum', hoặc 'none'
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        reduction="mean"
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits,
        targets
    ):
        """
        Args:
            logits: (batch_size, num_labels) - raw logits
            targets: (batch_size, num_labels) - binary targets (0 or 1)
        """
        targets = targets.float()

        # BCE loss per element (reduction='none')
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        # p_t: probability of correct prediction
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)

        # Focal modulating factor
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()

        if self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss