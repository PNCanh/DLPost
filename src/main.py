import argparse
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Config
from configs.model_config import CONFIGS, ACTIVE_CONFIG
from configs import paths

# Models and Layers
from models.multimodal.baseline_model import BaselineMultimodalModel
from models.multimodal.variant_a_model import VariantAModel
from models.multimodal.variant_b_model import VariantBModel
from losses.custom_losses import MultiTaskLoss
from trainers.adversarial import FGM
from trainers.layer_wise_lr import get_layer_wise_optimizer_grouped_parameters

# Data Pipelines
from data.ocr.ocr_pipeline import OCRPipeline
from data.builders.build_processed_data import build_dataset
from data.builders.build_image_dataset import build_image_dataset
from data.splitters.split_dataset import split_dataset
from data.dataset import load_train_dataset, load_val_dataset, load_test_dataset
from data.pytorch_datasets import get_dataloaders


def print_step(step_name):
    print("\n" + "=" * 60)
    print(f"  [*] {step_name}")
    print("=" * 60)


def get_kaggle_dir():
    """
    Tự động lấy đường dẫn Kaggle dataset.
    Ưu tiên biến môi trường KAGGLE_DATASET_DIR (được set trên Colab).
    Nếu không có, thử tải bằng kagglehub.
    """
    env_dir = os.environ.get("KAGGLE_DATASET_DIR")
    if env_dir and Path(env_dir).exists():
        print(f"[KAGGLE] Sử dụng dataset từ biến môi trường: {env_dir}")
        return env_dir

    try:
        import kagglehub
        print("[KAGGLE] Đang tải dataset bằng kagglehub...")
        kaggle_path = kagglehub.dataset_download(
            "cashbowman/ai-generated-images-vs-real-images"
        )
        print(f"[KAGGLE] Đã tải về: {kaggle_path}")
        return kaggle_path
    except Exception as e:
        print(f"[KAGGLE] Không thể tải dataset: {e}")
        return None


def train_epoch(model, loader, optimizer, criterion, device, adv_trainer=None):
    model.train()
    total_loss = 0
    total_bin_loss = 0
    total_cat_loss = 0
    total_exp_loss = 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)

        binary_label = batch["binary_label"].to(device)
        multi_label = batch["multi_label"].to(device)
        explanation_vector = batch["explanation_vector"].to(device)

        optimizer.zero_grad()

        preds_binary, preds_category, preds_explain = model(
            input_ids, attention_mask, images
        )

        loss, l_bin, l_cat, l_exp = criterion(
            preds_binary, preds_category, preds_explain,
            binary_label, multi_label, explanation_vector,
        )

        loss.backward()

        # Adversarial Training (FGM)
        if adv_trainer is not None:
            adv_trainer.attack()
            preds_adv = model(input_ids, attention_mask, images)
            loss_adv, _, _, _ = criterion(
                preds_adv[0], preds_adv[1], preds_adv[2],
                binary_label, multi_label, explanation_vector,
            )
            loss_adv.backward()
            adv_trainer.restore()

        optimizer.step()

        total_loss += loss.item()
        total_bin_loss += l_bin.item()
        total_cat_loss += l_cat.item()
        total_exp_loss += l_exp.item()

    n = len(loader)
    return total_loss / n, total_bin_loss / n, total_cat_loss / n, total_exp_loss / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_bin_loss = 0
    total_cat_loss = 0
    total_exp_loss = 0

    for batch in tqdm(loader, desc="Evaluation"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)

        binary_label = batch["binary_label"].to(device)
        multi_label = batch["multi_label"].to(device)
        explanation_vector = batch["explanation_vector"].to(device)

        preds_binary, preds_category, preds_explain = model(
            input_ids, attention_mask, images
        )

        loss, l_bin, l_cat, l_exp = criterion(
            preds_binary, preds_category, preds_explain,
            binary_label, multi_label, explanation_vector,
        )

        total_loss += loss.item()
        total_bin_loss += l_bin.item()
        total_cat_loss += l_cat.item()
        total_exp_loss += l_exp.item()

    n = len(loader)
    return total_loss / n, total_bin_loss / n, total_cat_loss / n, total_exp_loss / n


# =====================================================================
#  PIPELINE FUNCTIONS
# =====================================================================

def run_full_pipeline(config):
    """Chạy toàn bộ pipeline: OCR → Tiền xử lý → Build Image → Split → Train."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cấu hình: {config['run_name']} | Thiết bị: {device}")

    # --- 1. OCR ---
    if config["ocr"]["enabled"]:
        print_step("1/6  OCR – TRÍCH XUẤT VĂN BẢN TỪ ẢNH")
        ocr = OCRPipeline(
            languages=config["ocr"]["languages"],
            gpu=config["ocr"]["gpu"],
        )
        ocr.run_ocr_on_dataset()
    else:
        print_step("1/6  OCR – BỎ QUA (disabled trong config)")

    # --- 2. Tiền xử lý text ---
    print_step("2/6  TIỀN XỬ LÝ VĂN BẢN & XÂY DỰNG DATASET")
    build_dataset()

    # --- 3. Build Image Dataset (luôn kèm Kaggle) ---
    print_step("3/6  XÂY DỰNG IMAGE DATASET (post images + Kaggle)")
    kaggle_dir = get_kaggle_dir()
    build_image_dataset(kaggle_dir=kaggle_dir)

    # --- 4. Split ---
    print_step("4/6  CHIA TẬP DỮ LIỆU (70 / 15 / 15)")
    split_dataset()

    # --- 5 & 6: Train ---
    _train_model(config, device)


def run_train_only(config):
    """Chỉ chạy huấn luyện, data đã có sẵn trên GDrive."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cấu hình: {config['run_name']} | Thiết bị: {device}")
    _train_model(config, device)


def _train_model(config, device):
    """Logic huấn luyện dùng chung cho cả 2 option."""

    # Load data
    print_step("5/6  KHỞI TẠO MODEL & DATALOADER")
    try:
        train_df = load_train_dataset()
        val_df = load_val_dataset()
        test_df = load_test_dataset()
        print(f"Loaded  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    except Exception as e:
        print(f"[LỖI] Không load được dữ liệu: {e}")
        print("Hãy chạy full pipeline trước khi chạy train-only.")
        return

    train_loader, val_loader, test_loader = get_dataloaders(
        config, train_df, val_df, test_df
    )

    # Model
    model_type = config["model_type"]
    if model_type == "baseline":
        model = BaselineMultimodalModel()
    elif model_type == "variant_a":
        model = VariantAModel()
    elif model_type == "variant_b":
        model = VariantBModel()
    else:
        raise ValueError(f"model_type không hợp lệ: {model_type}")
    model = model.to(device)

    # Optimizer (Layer-wise LR nếu bật)
    if config["training"]["layer_wise_lr"]:
        print("Áp dụng Layer-wise Learning Rate Decay.")
        grouped = get_layer_wise_optimizer_grouped_parameters(
            model,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        optimizer = optim.AdamW(grouped)
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

    # Loss
    weights = [
        config["loss_weights"]["binary"],
        config["loss_weights"]["multiclass"],
        config["loss_weights"]["explanation"],
    ]
    criterion = MultiTaskLoss(weights=weights).to(device)

    # Adversarial
    adv_trainer = None
    if config["training"]["adversarial"] == "fgm":
        print("Khởi tạo Adversarial Training: FGM")
        adv_trainer = FGM(model, emb_name="word_embeddings")

    # Training loop
    print_step("6/6  HUẤN LUYỆN")
    best_val_loss = float("inf")
    epochs = config["training"]["epochs"]

    ckpt_dir = paths.OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        t_loss, t_bin, t_cat, t_exp = train_epoch(
            model, train_loader, optimizer, criterion, device, adv_trainer
        )
        print(
            f"[Train] Loss={t_loss:.4f}  "
            f"Binary={t_bin:.4f}  Category={t_cat:.4f}  Expl={t_exp:.4f}"
        )

        v_loss, v_bin, v_cat, v_exp = eval_epoch(
            model, val_loader, criterion, device
        )
        print(
            f"[Val]   Loss={v_loss:.4f}  "
            f"Binary={v_bin:.4f}  Category={v_cat:.4f}  Expl={v_exp:.4f}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_path = ckpt_dir / f"{config['run_name']}_best_{ts}.pth"
            torch.save(model.state_dict(), str(save_path))
            print(f"==> Lưu checkpoint tốt nhất: {save_path}")

    print("\n✅ HUẤN LUYỆN HOÀN TẤT!")

    # --- Evaluate on test set ---
    print_step("ĐÁNH GIÁ TRÊN TẬP TEST")
    predict_and_evaluate(model, test_loader, config, device, ts)


@torch.no_grad()
def predict_and_evaluate(model, test_loader, config, device, timestamp):
    """Chạy inference trên test set, tính metrics, lưu kết quả."""
    from evaluators.evaluator import ClassificationEvaluator

    model.eval()
    all_preds_binary = []
    all_preds_multi = []
    all_true_binary = []
    all_true_multi = []
    all_post_ids = []

    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)

        preds_bin, preds_cat, _ = model(input_ids, attention_mask, images)

        # Binary
        bin_probs = torch.sigmoid(preds_bin.squeeze(-1))
        bin_preds = (bin_probs > 0.5).long().cpu().tolist()
        all_preds_binary.extend(bin_preds)
        all_true_binary.extend(batch["binary_label"].long().tolist())

        # Multi-class
        cat_preds = preds_cat.argmax(dim=-1).cpu().tolist()
        all_preds_multi.extend(cat_preds)
        all_true_multi.extend(batch["multi_label"].tolist())

        # Post ids
        all_post_ids.extend(batch.get("post_id", [""] * len(bin_preds)))

    # Tạo predictions list cho evaluator
    predictions = []
    for i in range(len(all_preds_binary)):
        predictions.append({
            "post_id": all_post_ids[i],
            "true_label": all_true_multi[i],
            "predicted_label": all_preds_multi[i],
            "true_binary": all_true_binary[i],
            "predicted_binary": all_preds_binary[i],
            "is_scam": all_preds_binary[i] == 1,
        })

    model_name = f"{config.get('text_model', 'unknown')}_{config.get('image_model', 'unknown')}"
    evaluator = ClassificationEvaluator(
        run_name=config["run_name"],
        model_name=model_name,
        timestamp=timestamp,
    )

    training_metrics = {"best_val_loss": float("inf")}  # placeholder
    evaluator.evaluate(predictions, training_metrics)

    # In tóm tắt metrics
    from evaluators.metrics import compute_metrics
    binary_metrics = compute_metrics(all_true_binary, all_preds_binary)
    multi_metrics = compute_metrics(all_true_multi, all_preds_multi)

    print("\n📊 BINARY CLASSIFICATION:")
    for k, v in binary_metrics.items():
        print(f"   {k:12s}: {v:.4f}")

    print("\n📊 MULTICLASS CLASSIFICATION:")
    for k, v in multi_metrics.items():
        print(f"   {k:12s}: {v:.4f}")


# =====================================================================
#  ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DLPost – Multimodal Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=ACTIVE_CONFIG,
        help="Tên config (baseline | variant_a | variant_b)",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Chỉ train (bỏ qua OCR, tiền xử lý, split). Dữ liệu cần có sẵn.",
    )
    args = parser.parse_args()

    config = CONFIGS.get(args.config)
    if not config:
        raise ValueError(
            f"Config '{args.config}' không tồn tại. "
            f"Có sẵn: {list(CONFIGS.keys())}"
        )

    if args.train_only:
        run_train_only(config)
    else:
        run_full_pipeline(config)


if __name__ == "__main__":
    main()