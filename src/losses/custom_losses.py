import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class MultiTaskLoss(nn.Module):
    """
    Kết hợp 3 loại loss cho 3 task:
    1. Binary Classification -> BCEWithLogitsLoss
    2. Category Classification -> Focal Loss hoặc Weighted Cross Entropy
    3. Explanation -> BCEWithLogitsLoss (Multi-label)
    """
    def __init__(self, weights=[1.0, 1.0, 0.5]):
        super(MultiTaskLoss, self).__init__()
        self.weights = weights
        self.loss_binary = nn.BCEWithLogitsLoss()
        self.loss_category = FocalLoss(gamma=2.0)
        self.loss_explanation = nn.BCEWithLogitsLoss()

    def forward(self, preds_binary, preds_category, preds_explain, targets_binary, targets_category, targets_explain):
        # Ensure correct shapes
        preds_binary = preds_binary.squeeze(-1) if preds_binary.dim() > 1 else preds_binary
        targets_binary = targets_binary.float()
        
        l_bin = self.loss_binary(preds_binary, targets_binary)
        l_cat = self.loss_category(preds_category, targets_category)
        l_exp = self.loss_explanation(preds_explain, targets_explain.float())
        
        total_loss = self.weights[0] * l_bin + self.weights[1] * l_cat + self.weights[2] * l_exp
        return total_loss, l_bin, l_cat, l_exp
