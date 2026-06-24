import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from data.ocr.ocr_handle import OCRHandle
from data.builders.build_processed_data import build_dataset
from data.builders.build_image_dataset import build_image_dataset
from data.splitters.split_dataset import split_dataset

from data.dataset import load_train_dataset, load_val_dataset, load_test_dataset
from data.pytorch_datasets import get_dataloaders, get_image_dataloaders
from models.model_factory import ModelFactory
from losses.loss_factory import build_loss
from trainers.trainer_multimodal import MultimodalTrainer
from trainers.trainer_image import ImageTrainer
from evaluators.evaluator import ClassificationEvaluator
from configs import paths
from configs.model_config import TRAINING_CONFIGS, DEFAULT_CONFIGS
from evaluators.model_comparator import generate_comparison_report

def train_and_evaluate_single_config(config_name, config, train_df, val_df, test_df, image_train_df, image_val_df, image_test_df, device):
    print(f"\n{'='*50}")
    print(f"=== RUNNING CONFIG: {config_name} ===")
    print(f"{'='*50}")

    timestamp = datetime.now().strftime("%M%H%d%m%y")
    model_name = f"{config.get('text_model', 'text')}_{config.get('image_model', 'img')}_{config.get('fusion_strategy', 'fusion')}"
    config['run_timestamp'] = timestamp
    config['model_name'] = model_name

    train_loader, val_loader, test_loader = get_dataloaders(config, train_df, val_df, test_df)

    print("\n--- TRAIN MULTIMODAL MODEL ---")
    model = ModelFactory.build_model(config).to(device)
    loss_fn = build_loss(config["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    
    trainer = MultimodalTrainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        device=device, config=config
    )
    trainer.fit()

    print("\n--- EVALUATE MULTIMODAL MODEL ---")
    predictions = trainer.predict(test_loader)
    
    formatted_preds = []
    for i, item in test_df.iterrows():
        if i < len(predictions):
            pred_item = predictions[i]
            if isinstance(pred_item, dict) and "multiclass_probs" in pred_item:
                pred_label = int(np.argmax(pred_item["multiclass_probs"]))
            elif isinstance(pred_item, dict) and "binary_probs" in pred_item:
                pred_label = int(np.argmax(pred_item["binary_probs"]))
            else:
                pred_label = 0
                
            formatted_preds.append({
                "post_id": item.get("post_id", str(i)),
                "true_label": item.get("multi_label", 0),
                "predicted_label": pred_label,
                "explanation_probs": pred_item.get("explanation_probs", []) if isinstance(pred_item, dict) else [],
                "is_scam": pred_label != 0
            })

    # Giải phóng GPU memory của multimodal model để tránh tràn VRAM
    import gc
    del model
    del optimizer
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n--- TRAIN & EVALUATE INDEPENDENT IMAGE MODEL ---")
    print(f"Image Dataset sizes: Train={len(image_train_df)}, Val={len(image_val_df)}, Test={len(image_test_df)}")
    
    image_train_loader, image_val_loader, image_test_loader = get_image_dataloaders(
        config, image_train_df, image_val_df, image_test_df
    )
    
    image_model_name = config.get("image_model", "resnet")
    print(f"Building independent image model '{image_model_name}'...")
    if image_model_name == "resnet":
        print("Note: If running for the first time, this may download ~100MB ResNet50 weights from PyTorch Hub.")
        from models.image_models.resnet import ResNetClassifier
        image_model = ResNetClassifier(num_multiclass=8, num_explanations=10).to(device)
    else:
        print("Note: If running for the first time, this may download ~350MB ViT weights from Hugging Face.")
        from models.image_models.vit import ViTClassifier
        image_model = ViTClassifier(num_multiclass=8, num_explanations=10).to(device)
        
    image_optimizer = torch.optim.AdamW(image_model.parameters(), lr=config["training"]["learning_rate"])
    
    image_trainer = ImageTrainer(
        model=image_model, optimizer=image_optimizer, loss_fn=loss_fn,
        train_loader=image_train_loader, val_loader=image_val_loader,
        device=device, config=config
    )
    image_trainer.fit()
    image_predictions = image_trainer.predict(image_test_loader)
    
    image_preds_by_post = {}
    for i, row in image_test_df.iterrows():
        post_id = row.get("post_id", "")
        if post_id and i < len(image_predictions):
            pred_item = image_predictions[i]
            if isinstance(pred_item, dict) and "multiclass_probs" in pred_item:
                pred_label = int(np.argmax(pred_item["multiclass_probs"]))
                image_preds_by_post[post_id] = pred_label

    print("\n--- COMBINED RESULTS & EVALUATION ---")
    from data.builders.label_encoder import load_label_metadata
    from data.builders.explanation_encoder import load_explanation_mapping
    
    label_meta = load_label_metadata()
    explanation_mapping = load_explanation_mapping()
    id_to_explanation = {v: k for k, v in explanation_mapping.items()}
    
    for i, item in test_df.iterrows():
        post_id = item.get("post_id", str(i))
        text_pred = formatted_preds[i]
        text_label_id = text_pred["predicted_label"]
        text_label_name = label_meta.get(text_label_id, {}).get("name", str(text_label_id))
        
        img_label_str = "No Image"
        if item.get("has_image", 0) > 0 and post_id in image_preds_by_post:
            img_label_id = image_preds_by_post[post_id]
            img_label_str = label_meta.get(img_label_id, {}).get("name", str(img_label_id))
            
        explanation_probs = text_pred.get("explanation_probs", [])
        reasons = []
        if len(explanation_probs) > 0:
            for idx, prob in enumerate(explanation_probs):
                if prob > 0.5 and idx in id_to_explanation:
                    reasons.append(id_to_explanation[idx])
        
        reason_str = ", ".join(reasons) if reasons else "None"
        print(f"Post {post_id}: Text -> {text_label_name} | Image -> {img_label_str} | Reason -> {reason_str}")

    print("\n[Generating Classification Report]")
    run_name = config.get("run_name", "experiment")
    evaluator = ClassificationEvaluator(
        run_name=run_name,
        model_name=model_name,
        timestamp=timestamp
    )
    mock_metrics = {"train_loss": [], "val_loss": []}
    evaluator.evaluate(formatted_preds, mock_metrics)

    # Giải phóng GPU memory của image model
    del image_model
    del image_optimizer
    del image_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    
    # Tạo thư mục cần thiết
    paths.ensure_directories()
        
    print("\n=== STEP 1: EXTRACT TEXT VIA OCR ===")
    OCRHandle().process_all(paths.RAW_DIR)

    print("\n=== STEP 2: BUILD MULTIMODAL (TEXT) DATA ===")
    build_dataset()

    print("\n=== STEP 3: BUILD INDEPENDENT IMAGE DATA ===")
    build_image_dataset()

    print("\n=== STEP 4: SPLIT MULTIMODAL & IMAGE DATA ===")
    split_dataset()

    print("\n=== STEP 5: LOAD DATALOADERS & SET HF TOKEN ===")
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    else:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    train_df = load_train_dataset()
    val_df = load_val_dataset()
    test_df = load_test_dataset()
    
    image_train_df = pd.read_parquet(paths.IMAGE_TRAIN_PARQUET_FILE)
    image_val_df = pd.read_parquet(paths.IMAGE_VAL_PARQUET_FILE)
    image_test_df = pd.read_parquet(paths.IMAGE_TEST_PARQUET_FILE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chỉ train các configs được chọn trong DEFAULT_CONFIGS
    print(f"\n=== STEP 6: TRAIN SELECTED CONFIGS ({len(DEFAULT_CONFIGS)}) ===")
    print(f"Selected: {DEFAULT_CONFIGS}")
    
    for config_name in DEFAULT_CONFIGS:
        if config_name not in TRAINING_CONFIGS:
            print(f"⚠️  Config '{config_name}' not found in TRAINING_CONFIGS, skipping.")
            continue
        config = TRAINING_CONFIGS[config_name]
        train_and_evaluate_single_config(
            config_name, config, 
            train_df, val_df, test_df, 
            image_train_df, image_val_df, image_test_df, 
            device
        )
        
    print("\n=== STEP 7: GENERATE COMPARISON REPORT ===")
    generate_comparison_report()
    
    print("\nEND")

if __name__ == "__main__":
    main()