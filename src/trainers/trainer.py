import random

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_architecture = config.get("model_architecture")
        self.fusion_strategy = config.get("fusion_strategy")
        self.loss_function = config.get("loss_function")
        self.epochs = config.get("epochs", 1)
        
        # TODO: Initialize your model here using the config parameters.
        # Example:
        # self.model = build_model(
        #     architecture=self.model_architecture,
        #     fusion=self.fusion_strategy
        # )
        # self.criterion = get_loss(self.loss_function)
        # self.optimizer = get_optimizer(self.model, lr=config.get("learning_rate"))
        
    def train(self, train_data: list, val_data: list):
        """
        Train the model using train_data, and validate using val_data.
        
        Parameters:
            train_data (list): List of training examples (each example is a dict from train.json).
            val_data (list): List of validation examples (each example is a dict from val.json).
        """
        print(f"Training model [{self.config.get('run_name')}] for {self.epochs} epochs...")
        metrics = {"train_loss": [], "val_loss": []}
        
        # TODO: Implement your actual training loop here.
        # for epoch in range(self.epochs):
        #     # Train step
        #     # Validate step
        #     # Record metrics
        
        # MOCKING METRICS FOR PIPELINE COMPLETENESS
        for epoch in range(self.epochs):
            metrics["train_loss"].append(round(random.uniform(0.1, 0.5), 4))
            metrics["val_loss"].append(round(random.uniform(0.15, 0.6), 4))
            
        print("Training completed.")
        return metrics

    def predict(self, test_data: list):
        """
        Run inference on the test_data.
        
        Parameters:
            test_data (list): List of test examples (from test.json).
            
        Returns:
            list: List of dictionaries containing "post_id", "true_label", and "predicted_label".
        """
        predictions = []
        
        # TODO: Implement actual inference
        # preds = self.model(test_data)
        
        # MOCKING PREDICTIONS FOR PIPELINE COMPLETENESS
        for item in test_data:
            true_label = item.get("binary_label", 0)
            # 80% chance to be correct
            pred_label = true_label if random.random() < 0.8 else 1 - true_label
            
            predictions.append({
                "post_id": item.get("post_id"),
                "true_label": true_label,
                "predicted_label": pred_label
            })
            
        return predictions
from abc import (
    ABC,
    abstractmethod
)

from pathlib import Path

import torch
from configs.paths import CHECKPOINTS_DIR


class BaseTrainer(ABC):

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

        self.model = model

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.device = device

        self.config = config

        self.num_epochs = config[
            "training"
        ][
            "epochs"
        ]

        timestamp = self.config.get("run_timestamp", "0000000000")
        model_name = self.config.get("model_name", "model")

        self.checkpoint_dir = CHECKPOINTS_DIR / f"{model_name}_{timestamp}"

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        
        # Setup logging file
        from datetime import datetime
        now_str = datetime.now().strftime("%m%H%M%d%y")
        # Go up from src/trainers/trainer.py -> src/trainers -> src -> DLPost
        project_dir = Path(__file__).resolve().parents[2]
        self.logs_dir = project_dir / "outputs" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / f"{model_name}_{now_str}.log"
        
        # Write header to log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Timestamp: {now_str}\n")
            f.write("-" * 30 + "\n")

        self.best_val_loss = float(
            "inf"
        )

        self.patience = config[
            "training"
        ].get(
            "early_stopping",
            5
        )

        self.patience_counter = 0

        self.use_amp = config[
            "training"
        ].get(
            "mixed_precision",
            True
        )

        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.use_amp
            else None
        )

    def fit(self):

        for epoch in range(
            1,
            self.num_epochs + 1
        ):

            train_loss = (
                self.train_epoch()
            )

            val_loss = (
                self.validate_epoch()
            )
            
            # Log to terminal
            print(
                f"\nEpoch "
                f"{epoch}/"
                f"{self.num_epochs}"
            )

            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Val Loss: "
                f"{val_loss:.4f}"
            )
            
            # Write to log file
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch}/{self.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}\n")

            if (
                val_loss
                < self.best_val_loss
            ):

                self.best_val_loss = (
                    val_loss
                )

                self.patience_counter = 0

                self.save_checkpoint(
                    "best_model.pt"
                )

            else:

                self.patience_counter += 1

            if (
                self.patience_counter
                >= self.patience
            ):

                print(
                    "\nEarly stopping."
                )
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write("Early stopping triggered.\n")

                break

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0

        for batch in (
            self.train_loader
        ):

            self.optimizer.zero_grad()

            if self.use_amp:

                with torch.amp.autocast('cuda'):

                    loss_dict = (
                        self._compute_loss(
                            batch
                        )
                    )
                    loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict

                self.scaler.scale(
                    loss
                ).backward()

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:

                loss_dict = (
                    self._compute_loss(
                        batch
                    )
                )
                loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict

                loss.backward()

                self.optimizer.step()

            total_loss += (
                loss.item()
            )

        return (

            total_loss
            / len(
                self.train_loader
            )
        )

    def validate_epoch(self):

        self.model.eval()

        total_loss = 0.0

        with torch.no_grad():

            for batch in (
                self.val_loader
            ):

                loss_dict = (
                    self._compute_loss(
                        batch
                    )
                )
                loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict

                total_loss += (
                    loss.item()
                )

        return (

            total_loss
            / len(
                self.val_loader
            )
        )

    def save_checkpoint(
        self,
        filename
    ):
        timestamp = self.config.get("run_timestamp", "0000000000")
        model_name = self.config.get("model_name", "model")
        actual_filename = f"{model_name}_{timestamp}_{filename}"
        checkpoint_path = (

            self.checkpoint_dir
            / actual_filename
        )

        torch.save(

            {

                "model_state_dict":
                    self.model.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "best_val_loss":
                    self.best_val_loss
            },

            checkpoint_path
        )

        print(
            f"Saved: "
            f"{checkpoint_path}"
        )

    def load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint = torch.load(

            checkpoint_path,

            map_location=self.device
        )

        self.model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(

            checkpoint[
                "optimizer_state_dict"
            ]
        )

        self.best_val_loss = (

            checkpoint[
                "best_val_loss"
            ]
        )

        print(
            f"Loaded: "
            f"{checkpoint_path}"
        )

    def predict(
        self,
        loader
    ):

        self.model.eval()

        outputs = []

        with torch.no_grad():

            for batch in loader:

                predictions = (
                    self._predict(
                        batch
                    )
                )
                
                # Chuyển predictions từ dict of tensors sang list of dicts (per sample)
                batch_size = predictions["binary_probs"].shape[0]
                for i in range(batch_size):
                    item_pred = {
                        "binary_probs": predictions["binary_probs"][i].cpu().numpy(),
                        "multiclass_probs": predictions["multiclass_probs"][i].cpu().numpy(),
                        "explanation_probs": predictions["explanation_probs"][i].cpu().numpy()
                    }
                    outputs.append(item_pred)

        return outputs

    def _extract_targets(
    self,
    batch
    ):

        return {

        "binary_label":
            batch[
                "binary_label"
            ],

        "multi_label":
            batch[
                "multi_label"
            ],

        "explanation_vector":
            batch[
                "explanation_vector"
            ]
    }

    @abstractmethod
    def _compute_loss(
        self,
        batch
    ):
        pass

    @abstractmethod
    def _predict(
        self,
        batch
    ):
        pass