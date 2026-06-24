import torch

from trainers.trainer import (
    BaseTrainer
)


class ImageTrainer(
    BaseTrainer
):

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        device,
        config
    ):

        super().__init__(
            model,
            optimizer,
            loss_fn,
            train_loader,
            val_loader,
            device,
            config
        )

    def _move_to_device(
        self,
        batch
    ):

        return {

            k:
            v.to(self.device)

            if isinstance(
                v,
                torch.Tensor
            )

            else v

            for k, v in batch.items()
        }

    def _compute_loss(
        self,
        batch
    ):
        batch = self._move_to_device(
            batch
        )
        outputs = self.model(
            batch["image"]
        )
        return self.loss_fn(
            outputs,
            self._extract_targets(
                batch
            )
        )

    def _predict(
        self,
        batch
    ):
        batch = self._move_to_device(
            batch
        )
        outputs = self.model(
            batch["image"]
        )
        return self._build_predictions(
            outputs
        )

    def _build_predictions(
        self,
        outputs
    ):

        binary_probs = torch.sigmoid(

            outputs[
                "binary_logits"
            ]
        )

        multiclass_probs = torch.softmax(

            outputs[
                "multiclass_logits"
            ],

            dim=1
        )

        explanation_probs = torch.sigmoid(

            outputs[
                "explanation_logits"
            ]
        )

        return {

            "binary_probs":
                binary_probs,

            "multiclass_probs":
                multiclass_probs,

            "explanation_probs":
                explanation_probs
        }