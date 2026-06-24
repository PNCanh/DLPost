import torch.nn as nn

from losses.focal_loss import (
    FocalLoss
)

from losses.multitask_loss import (
    MultiTaskLoss
)


def build_single_loss(
    loss_config
):

    loss_type = loss_config["type"]

    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    if loss_type == "weighted_ce":
        # Extract weights from config or use None.
        # In practice, these weights should be passed dynamically, 
        # but for demonstration we initialize with dummy or config weights.
        weights = loss_config.get("class_weights", None)
        if weights is not None:
            import torch
            weights = torch.tensor(weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)

    if loss_type == "focal":
        return FocalLoss()

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_type == "weighted_bce":
        weights = loss_config.get("pos_weights", None)
        if weights is not None:
            import torch
            weights = torch.tensor(weights, dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=weights)

    raise ValueError(
        f"Unknown loss: {loss_type}"
    )


def build_loss(
    config
):

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

    return MultiTaskLoss(

        binary_loss_fn=
            binary_loss_fn,

        multiclass_loss_fn=
            multiclass_loss_fn,

        explanation_loss_fn=
            explanation_loss_fn,

        binary_weight=
            config[
                "weights"
            ]["binary"],

        multiclass_weight=
            config[
                "weights"
            ]["multiclass"],

        explanation_weight=
            config[
                "weights"
            ]["explanation"]
    )