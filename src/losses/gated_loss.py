"""
Gated Loss (Nhóm 2 - 1 điểm)

Nếu aspect không được mention (binary_label == 0, tức legitimate),
thì không tính loss sentiment (multiclass + explanation) của aspect đó.

Logic:
- Tạo mask từ binary_label: mask = (binary_label != 0).float()
- multiclass_loss = multiclass_loss * mask.mean()  (chỉ tính trên samples scam)
- explanation_loss = explanation_loss * mask.mean() (chỉ tính trên samples scam)
- binary_loss vẫn tính bình thường (cần phân biệt scam/legit cho mọi sample)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiTaskLoss(nn.Module):
    """
    Multitask loss với gating mechanism.
    
    Khi binary_label == 0 (legitimate):
    - multiclass_loss bị mask (= 0) vì không cần phân loại chi tiết
    - explanation_loss bị mask (= 0) vì không có lý do scam
    
    Khi binary_label == 1 (scam):
    - Tất cả loss đều được tính bình thường
    """

    def __init__(
        self,
        binary_loss_fn,
        multiclass_loss_fn,
        explanation_loss_fn,
        binary_weight=1.0,
        multiclass_weight=1.0,
        explanation_weight=1.0
    ):

        super().__init__()

        self.binary_loss_fn = binary_loss_fn
        self.multiclass_loss_fn = multiclass_loss_fn
        self.explanation_loss_fn = explanation_loss_fn

        self.binary_weight = binary_weight
        self.multiclass_weight = multiclass_weight
        self.explanation_weight = explanation_weight

    def forward(self, outputs, targets):
        # === Binary Loss (luôn tính) ===
        binary_logits = outputs["binary_logits"]
        binary_target = targets["binary_label"]

        if isinstance(self.binary_loss_fn, nn.BCEWithLogitsLoss):
            if binary_logits.shape[-1] == 2:
                binary_target_for_loss = F.one_hot(
                    binary_target.long(), num_classes=2
                ).float()
            else:
                binary_target_for_loss = binary_target.float()
        else:
            binary_target_for_loss = binary_target.long()

        binary_loss = self.binary_loss_fn(
            binary_logits,
            binary_target_for_loss
        )

        # === Tạo gate mask từ binary_label ===
        # mask = 1 nếu scam (binary_label != 0), 0 nếu legitimate
        gate_mask = (targets["binary_label"] != 0).float()

        # === Multiclass Loss (chỉ tính cho scam samples) ===
        multiclass_logits = outputs["multiclass_logits"]
        multiclass_target = targets["multi_label"]

        if gate_mask.sum() > 0:
            # Tính loss per-sample (reduction='none')
            multiclass_loss_per_sample = F.cross_entropy(
                multiclass_logits,
                multiclass_target.long(),
                reduction='none'
            )
            # Áp dụng mask: chỉ tính loss cho scam samples
            multiclass_loss = (multiclass_loss_per_sample * gate_mask).sum() / gate_mask.sum()
        else:
            multiclass_loss = torch.tensor(0.0, device=binary_logits.device)

        # === Explanation Loss (chỉ tính cho scam samples) ===
        explanation_logits = outputs["explanation_logits"]
        explanation_target = targets["explanation_vector"]

        if gate_mask.sum() > 0:
            # Tính BCE loss per-sample
            explanation_loss_per_sample = F.binary_cross_entropy_with_logits(
                explanation_logits,
                explanation_target.float(),
                reduction='none'
            ).mean(dim=1)  # mean qua các explanation dimensions
            # Áp dụng mask
            explanation_loss = (explanation_loss_per_sample * gate_mask).sum() / gate_mask.sum()
        else:
            explanation_loss = torch.tensor(0.0, device=binary_logits.device)

        # === Total Loss ===
        total_loss = (
            self.binary_weight * binary_loss
            + self.multiclass_weight * multiclass_loss
            + self.explanation_weight * explanation_loss
        )

        return {
            "loss": total_loss,
            "binary_loss": binary_loss,
            "multiclass_loss": multiclass_loss,
            "explanation_loss": explanation_loss
        }
