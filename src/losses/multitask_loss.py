import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):

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

        self.binary_loss_fn = (
            binary_loss_fn
        )

        self.multiclass_loss_fn = (
            multiclass_loss_fn
        )

        self.explanation_loss_fn = (
            explanation_loss_fn
        )

        self.binary_weight = (
            binary_weight
        )

        self.multiclass_weight = (
            multiclass_weight
        )

        self.explanation_weight = (
            explanation_weight
        )

    def forward(
        self,
        outputs,
        targets
    ):

        binary_logits = outputs["binary_logits"]
        binary_target = targets["binary_label"]
        
        if isinstance(self.binary_loss_fn, nn.BCEWithLogitsLoss):
            if binary_logits.shape[-1] == 2:
                binary_target = F.one_hot(binary_target.long(), num_classes=2).float()
        else:
            binary_target = binary_target.long()

        binary_loss = self.binary_loss_fn(
            binary_logits,
            binary_target
        )

        multiclass_loss = (

            self.multiclass_loss_fn(

                outputs[
                    "multiclass_logits"
                ],

                targets[
                    "multi_label"
                ]
            )
        )

        explanation_loss = (

            self.explanation_loss_fn(

                outputs[
                    "explanation_logits"
                ],

                targets[
                    "explanation_vector"
                ]
            )
        )

        total_loss = (

            self.binary_weight
            * binary_loss

            +

            self.multiclass_weight
            * multiclass_loss

            +

            self.explanation_weight
            * explanation_loss
        )

        return {

            "loss":
                total_loss,

            "binary_loss":
                binary_loss,

            "multiclass_loss":
                multiclass_loss,

            "explanation_loss":
                explanation_loss
        }