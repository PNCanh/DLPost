"""
Weighted Loss (Nhóm 2 - 1 điểm)

Gán trọng số thấp hơn cho các nhãn có độ tin cậy thấp (risk_level thấp).

Dựa trên risk_level từ label metadata:
- legitimate: risk_level = 0 → weight thấp nhất
- suspicious: risk_level = 2 → weight thấp
- fake_image: risk_level = 3 → weight trung bình
- sale_scam, fake_course: risk_level = 4 → weight cao
- job_scam, investment_scam, prize_scam: risk_level = 5 → weight cao nhất

Class weights được tính: weight_i = risk_level_i / max_risk_level + base_weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Risk levels cho từng class (từ label metadata)
DEFAULT_RISK_LEVELS = {
    0: 0,  # legitimate
    1: 5,  # job_scam
    2: 5,  # investment_scam
    3: 4,  # sale_scam
    4: 3,  # fake_image
    5: 5,  # prize_scam
    6: 4,  # fake_course
    7: 2,  # suspicious
}


def compute_risk_weights(risk_levels=None, base_weight=0.3, num_classes=8):
    """
    Tính class weights từ risk_level.
    
    Args:
        risk_levels: Dict {class_id: risk_level}
        base_weight: Trọng số cơ bản cho mọi class (tránh weight = 0)
        num_classes: Số lượng classes
        
    Returns:
        torch.Tensor: Class weights (num_classes,)
    """
    if risk_levels is None:
        risk_levels = DEFAULT_RISK_LEVELS

    max_risk = max(risk_levels.values()) if risk_levels else 1
    if max_risk == 0:
        max_risk = 1

    weights = []
    for i in range(num_classes):
        risk = risk_levels.get(i, 1)
        # weight = base + (risk / max_risk) * (1 - base)
        w = base_weight + (risk / max_risk) * (1 - base_weight)
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


class WeightedCrossEntropyLoss(nn.Module):
    """
    CrossEntropy Loss với class weights dựa trên risk_level.
    
    Nhãn có risk_level cao (vd: investment_scam = 5) → weight cao → loss quan trọng hơn.
    Nhãn có risk_level thấp (vd: suspicious = 2) → weight thấp → loss ít ảnh hưởng.
    """

    def __init__(self, class_weights=None, risk_levels=None, num_classes=8):
        super().__init__()

        if class_weights is not None:
            # Dùng weights được truyền trực tiếp
            self.weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            # Tự tính từ risk_levels
            self.weights = compute_risk_weights(risk_levels, num_classes=num_classes)

        self.register_buffer('class_weights', self.weights)

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,) - class indices
        """
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=self.class_weights.to(logits.device)
        )


class WeightedBCELoss(nn.Module):
    """
    BCE Loss với trọng số theo độ tin cậy của từng explanation label.
    
    Mỗi explanation có thể có confidence khác nhau,
    labels có confidence thấp → weight thấp hơn.
    """

    def __init__(self, pos_weights=None, confidence_weights=None, num_labels=10):
        super().__init__()

        if pos_weights is not None:
            self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        else:
            self.pos_weights = torch.ones(num_labels, dtype=torch.float32)

        if confidence_weights is not None:
            self.confidence_weights = torch.tensor(confidence_weights, dtype=torch.float32)
        else:
            self.confidence_weights = torch.ones(num_labels, dtype=torch.float32)

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_labels)
            targets: (batch_size, num_labels)
        """
        # BCE with pos_weight
        loss_per_label = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weights.to(logits.device),
            reduction='none'
        )

        # Áp dụng confidence weights cho từng label
        conf_weights = self.confidence_weights.to(logits.device)
        weighted_loss = loss_per_label * conf_weights.unsqueeze(0)

        return weighted_loss.mean()
