import argparse
import torch
import torch.optim as optim
import pandas as pd
from datetime import datetime

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

# Pipelines
from data.ocr.ocr_pipeline import OCRPipeline
from data.builders.build_processed_data import process_raw_data # Giả sử có hàm build_processed_data xử lý toàn bộ

def print_step(step_name):
    print("\n" + "="*50)
    print(f"[*] STEP: {step_name}")
    print("="*50)

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
        
    print(f"Đang chạy với cấu hình: {config['run_name']}")
    
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
    # 2. Tiền xử lý dữ liệu (Text Cleaning, v.v.)
    # ==========================================
    if not args.skip_preprocess:
        print_step("PREPROCESSING DATA")
        print("Đang tiền xử lý text (thường, icon, teencode...) và tổng hợp metadata...")
        # Ở đây sẽ gọi hàm build_processed_data (nếu đã implement ở build_processed_data.py)
        # process_raw_data(paths.RAW_DIR, paths.PROCESSED_DATASET_FILE)
        print("Tiền xử lý hoàn tất! (Mock)")
    else:
        print_step("SKIPPING PREPROCESSING")

    # ==========================================
    # 3. Khởi tạo Mô hình
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

    # ==========================================
    # 4. Huấn luyện (Train)
    # ==========================================
    print_step("SETTING UP TRAINING")
    
    # Layer-wise LR
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
    criterion = MultiTaskLoss(weights=weights)

    # Adversarial Training
    adv_trainer = None
    if config['training']['adversarial'] == "fgm":
        print("Khởi tạo Adversarial Training: FGM")
        adv_trainer = FGM(model)

    print("Mô hình, Optimizer, Loss đã sẵn sàng. Bắt đầu vòng lặp huấn luyện...")
    # TODO: Khởi tạo DataLoader và viết Training Loop ở đây
    print(f"Huấn luyện trong {config['training']['epochs']} epochs với batch size {config['training']['batch_size']} (Mock)")

    # ==========================================
    # 5. Xuất kết quả (Export Output)
    # ==========================================
    print_step("EXPORTING OUTPUT")
    # Lấy đường dẫn outputs từ paths.py
    output_dir = paths.OUTPUT_DIR / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"{config['run_name']}_{timestamp}.pth"
    
    # Giả lập lưu mô hình (chỉ lưu state_dict của mô hình rỗng hiện tại)
    torch.save(model.state_dict(), str(model_path))
    print(f"Đã lưu mô hình tại: {model_path}")
    print("\nPIPELINE HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    main()