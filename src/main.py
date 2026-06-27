import argparse
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
from data.splitters.split_dataset import split_dataset
from data.dataset import load_train_dataset, load_val_dataset, load_test_dataset
from data.pytorch_datasets import get_dataloaders

def print_step(step_name):
    print("\n" + "="*50)
    print(f"[*] STEP: {step_name}")
    print("="*50)

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
        
        # Targets
        binary_label = batch["binary_label"].to(device)
        multi_label = batch["multi_label"].to(device)
        explanation_vector = batch["explanation_vector"].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        preds_binary, preds_category, preds_explain = model(input_ids, attention_mask, images)
        
        # Loss
        loss, l_bin, l_cat, l_exp = criterion(
            preds_binary, preds_category, preds_explain,
            binary_label, multi_label, explanation_vector
        )
        
        loss.backward()
        
        # Adversarial Training (FGM)
        if adv_trainer is not None:
            adv_trainer.attack()
            preds_binary_adv, preds_category_adv, preds_explain_adv = model(input_ids, attention_mask, images)
            loss_adv, _, _, _ = criterion(
                preds_binary_adv, preds_category_adv, preds_explain_adv,
                binary_label, multi_label, explanation_vector
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
        
        preds_binary, preds_category, preds_explain = model(input_ids, attention_mask, images)
        
        loss, l_bin, l_cat, l_exp = criterion(
            preds_binary, preds_category, preds_explain,
            binary_label, multi_label, explanation_vector
        )
        
        total_loss += loss.item()
        total_bin_loss += l_bin.item()
        total_cat_loss += l_cat.item()
        total_exp_loss += l_exp.item()
        
    n = len(loader)
    return total_loss / n, total_bin_loss / n, total_cat_loss / n, total_exp_loss / n

def main():
    parser = argparse.ArgumentParser(description="Multimodal Deep Learning Training Pipeline")
    parser.add_argument("--config", type=str, default=ACTIVE_CONFIG, help="Tên config trong model_config.py")
    parser.add_argument("--skip_ocr", action="store_true", help="Bỏ qua bước OCR")
    parser.add_argument("--skip_preprocess", action="store_true", help="Bỏ qua bước tiền xử lý (dùng data đã xử lý)")
    args = parser.parse_args()

    # Tải cấu hình
    config = CONFIGS.get(args.config)
    if not config:
        raise ValueError(f"Config '{args.config}' không tồn tại trong model_config.py!")
        
    print(f"Đang chạy cấu hình: {config['run_name']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # ==========================================
    # 1. OCR (Trích xuất văn bản từ hình ảnh)
    # ==========================================
    if config['ocr']['enabled'] and not args.skip_ocr:
        print_step("RUNNING OCR PIPELINE")
        ocr = OCRPipeline(languages=config['ocr']['languages'], gpu=config['ocr']['gpu'])
        ocr.run_ocr_on_dataset()
    else:
        print_step("SKIPPING OCR PIPELINE")

    # ==========================================
    # 2. Tiền xử lý dữ liệu (Text Cleaning & Split)
    # ==========================================
    if not args.skip_preprocess:
        print_step("PREPROCESSING & SPLITTING DATA")
        build_dataset()
        split_dataset()
    else:
        print_step("SKIPPING PREPROCESSING & SPLITTING")

    # ==========================================
    # 3. Load DataFrames
    # ==========================================
    print_step("LOADING DATASETS")
    try:
        train_df = load_train_dataset()
        val_df = load_val_dataset()
        test_df = load_test_dataset()
        print(f"Loaded train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    except Exception as e:
        print(f"Lỗi khi load dữ liệu: {e}. Vui lòng không dùng --skip_preprocess nếu chưa có dữ liệu.")
        return

    # ==========================================
    # 4. Khởi tạo Dataloaders
    # ==========================================
    print_step("CREATING DATALOADERS")
    train_loader, val_loader, test_loader = get_dataloaders(config, train_df, val_df, test_df)

    # ==========================================
    # 5. Khởi tạo Mô hình
    # ==========================================
    print_step(f"INITIALIZING MODEL: {config['model_type']}")
    if config['model_type'] == "baseline":
        model = BaselineMultimodalModel()
    elif config['model_type'] == "variant_a":
        model = VariantAModel()
    elif config['model_type'] == "variant_b":
        model = VariantBModel()
    else:
        raise ValueError("Model type không hợp lệ!")
    
    model = model.to(device)

    # ==========================================
    # 6. Setup Optimizer, Loss, Adversarial
    # ==========================================
    print_step("SETTING UP TRAINING COMPONENTS")
    
    # Optimizer & LLRD
    if config['training']['layer_wise_lr']:
        print("Áp dụng Layer-wise Learning Rate Decay.")
        optimizer_grouped_parameters = get_layer_wise_optimizer_grouped_parameters(
            model, 
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        optimizer = optim.AdamW(optimizer_grouped_parameters)
    else:
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate'], 
            weight_decay=config['training']['weight_decay']
        )

    # Multi-task Loss
    weights = [
        config['loss_weights']['binary'],
        config['loss_weights']['multiclass'],
        config['loss_weights']['explanation']
    ]
    criterion = MultiTaskLoss(weights=weights).to(device)

    # Adversarial Training (FGM)
    adv_trainer = None
    if config['training']['adversarial'] == "fgm":
        print("Khởi tạo Adversarial Training: FGM")
        # Tìm embeddings name thích hợp tùy model
        emb_name = 'word_embeddings'
        adv_trainer = FGM(model, emb_name=emb_name)

    # ==========================================
    # 7. Training Loop
    # ==========================================
    print_step("STARTING TRAINING LOOP")
    best_val_loss = float('inf')
    epochs = config['training']['epochs']
    
    output_dir = paths.OUTPUT_DIR / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        train_loss, train_bin, train_cat, train_exp = train_epoch(
            model, train_loader, optimizer, criterion, device, adv_trainer
        )
        print(f"[Train] Loss={train_loss:.4f} | Binary={train_bin:.4f} | Category={train_cat:.4f} | Expl={train_exp:.4f}")
        
        val_loss, val_bin, val_cat, val_exp = eval_epoch(
            model, val_loader, criterion, device
        )
        print(f"[Val] Loss={val_loss:.4f} | Binary={val_bin:.4f} | Category={val_cat:.4f} | Expl={val_exp:.4f}")
        
        # Lưu checkpoint tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = output_dir / f"{config['run_name']}_best_{timestamp}.pth"
            torch.save(model.state_dict(), str(model_path))
            print(f"==> Đã lưu mô hình tốt nhất mới tại: {model_path}")
            
    print("\nHUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    main()