"""
Model Configuration

Định nghĩa các cấu hình cho các pipeline huấn luyện và mô hình.
Người dùng có thể chọn cấu hình mong muốn để chạy trong main.py
"""

CONFIGS = {
    "baseline": {
        "run_name": "baseline_phobert_resnet",
        "model_type": "baseline",
        "text_model": "phobert",
        "image_model": "resnet",
        
        "ocr": {
            "enabled": True,
            "languages": ["vi", "en"],
            "gpu": True
        },
        
        "training": {
            "batch_size": 16,
            "learning_rate": 2e-5,
            "epochs": 5,
            "adversarial": "none", # "none", "fgm", "fgd", "awp"
            "weight_decay": 0.01,
            "layer_wise_lr": True,
        },
        
        "loss_weights": {
            "binary": 1.0,
            "multiclass": 1.0,
            "explanation": 0.5
        }
    },
    
    "variant_a": {
        "run_name": "variant_a_xlmr_vit",
        "model_type": "variant_a",
        "text_model": "xlmr",
        "image_model": "vit",
        
        "ocr": {
            "enabled": True,
            "languages": ["vi", "en"],
            "gpu": True
        },
        
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-5,
            "epochs": 5,
            "adversarial": "fgm",
            "weight_decay": 0.01,
            "layer_wise_lr": True,
        },
        
        "loss_weights": {
            "binary": 1.0,
            "multiclass": 1.5,
            "explanation": 1.0
        }
    },
    
    "variant_b": {
        "run_name": "variant_b_vibert_resnet",
        "model_type": "variant_b",
        "text_model": "vibert",
        "image_model": "resnet",
        
        "ocr": {
            "enabled": True,
            "languages": ["vi", "en"],
            "gpu": True
        },
        
        "training": {
            "batch_size": 16,
            "learning_rate": 2e-5,
            "epochs": 5,
            "adversarial": "none",
            "weight_decay": 0.01,
            "layer_wise_lr": True,
        },
        
        "loss_weights": {
            "binary": 1.0,
            "multiclass": 1.0,
            "explanation": 0.5
        }
    }
}

# Chọn config active để chạy
ACTIVE_CONFIG = "baseline"
