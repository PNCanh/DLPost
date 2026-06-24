"""
Loss Factory

Build loss functions dựa trên cấu hình trong model_config.py.

Supported loss types:
- "ce": CrossEntropyLoss (multiclass)
- "weighted_ce": CrossEntropyLoss với class weights
- "risk_weighted_ce": CrossEntropyLoss với weights tự tính từ risk_level
- "focal": FocalLoss cho multiclass
- "bce": BCEWithLogitsLoss (binary/multi-label)
- "weighted_bce": BCEWithLogitsLoss với pos_weight
- "binary_focal": BinaryFocalLoss
- "gated": GatedMultiTaskLoss (special - builds full multitask loss)
"""

import torch
import torch.nn as nn

from losses.focal_loss import (
    FocalLoss,
    BinaryFocalLoss
)

from losses.multitask_loss import (
    MultiTaskLoss
)

from losses.gated_loss import (
    GatedMultiTaskLoss
)

from losses.weighted_loss import (
    WeightedCrossEntropyLoss,
    WeightedBCELoss
)


def build_single_loss(
    loss_config
):
    """
    Build một loss function đơn lẻ từ config.
    
    Args:
        loss_config: dict với key "type" và các params tùy chọn
        
    Returns:
        nn.Module: Loss function
    """

    loss_type = loss_config["type"]

    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    if loss_type == "weighted_ce":
        weights = loss_config.get("class_weights", None)
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)

    if loss_type == "risk_weighted_ce":
        # Tự tính weights từ risk_level
        return WeightedCrossEntropyLoss(
            risk_levels=loss_config.get("risk_levels", None),
            num_classes=loss_config.get("num_classes", 8)
        )

    if loss_type == "focal":
        return FocalLoss(
            alpha=loss_config.get("alpha", 1.0),
            gamma=loss_config.get("gamma", 2.0)
        )

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_type == "weighted_bce":
        weights = loss_config.get("pos_weights", None)
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32)
        confidence_weights = loss_config.get("confidence_weights", None)
        if confidence_weights is not None:
            return WeightedBCELoss(
                pos_weights=loss_config.get("pos_weights", None),
                confidence_weights=confidence_weights
            )
        return nn.BCEWithLogitsLoss(pos_weight=weights)

    if loss_type == "binary_focal":
        return BinaryFocalLoss(
            alpha=loss_config.get("alpha", 0.25),
            gamma=loss_config.get("gamma", 2.0)
        )

    raise ValueError(
        f"Unknown loss: {loss_type}"
    )


def build_loss(config):
    """
    Build multitask loss từ config.
    
    Args:
        config: dict chứa config cho binary, multiclass, explanation losses
        
    Returns:
        nn.Module: MultiTaskLoss hoặc GatedMultiTaskLoss
    """

    # Kiểm tra xem có dùng gated loss không
    use_gated = config.get("gated", False)

    binary_loss_fn = (
        build_single_loss(
            config["binary"]
        )
    )

    multiclass_loss_fn = (
        build_single_loss(
            config["multiclass"]
        )
    )

    explanation_loss_fn = (
        build_single_loss(
            config["explanation"]
        )
    )

    weights = config.get("weights", {})

    # Chọn MultiTaskLoss class
    LossClass = GatedMultiTaskLoss if use_gated else MultiTaskLoss

    return LossClass(

        binary_loss_fn=
            binary_loss_fn,

        multiclass_loss_fn=
            multiclass_loss_fn,

        explanation_loss_fn=
            explanation_loss_fn,

        binary_weight=
            weights.get("binary", 1.0),

        multiclass_weight=
            weights.get("multiclass", 1.0),

        explanation_weight=
            weights.get("explanation", 1.0)
    )