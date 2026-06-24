"""
Model Configurations

Định nghĩa cấu hình chi tiết cho các components của models, losses, optimizer và fusion.
- text_model: Tên các mô hình ngôn ngữ (phobert, vibert, xlmr)
- image_model: Tên các mô hình ảnh (resnet, vit)
- fusion_strategy: Tên phương pháp nối đặc trưng (concat, attention, gated, conditional_attention)
"""

# Cấu hình đầy đủ cho một pipeline multimodal
TRAINING_CONFIGS = {
    "baseline": {
        "run_name": "baseline_attention",
        "text_model": "phobert",
        "image_model": "resnet",
        "fusion_strategy": "attention",
        "loss": {
            "binary": {"type": "bce"},
            "multiclass": {"type": "ce"},
            "explanation": {"type": "bce"},
            "weights": {
                "binary": 1.0,
                "multiclass": 1.0,
                "explanation": 0.5
            }
        },
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 4,
            "epochs": 10,
            "early_stopping": 5,
            "mixed_precision": True,
            "checkpoint_dir": "checkpoints"
        }
    },
    
    "modelSetting1": {
        "run_name": "vibert_vit_gated_focal",
        "text_model": "vibert",
        "image_model": "vit",
        "fusion_strategy": "gated",
        "loss": {
            "binary": {"type": "focal"},
            "multiclass": {"type": "focal"},
            "explanation": {"type": "weighted_bce", "pos_weights": [0.8] * 10}, # Example weights
            "weights": {
                "binary": 1.0,
                "multiclass": 1.5,
                "explanation": 1.0
            }
        },
        "training": {
            "learning_rate": 1e-5,
            "batch_size": 4,
            "epochs": 10,
            "early_stopping": 5,
            "mixed_precision": True,
            "checkpoint_dir": "checkpoints"
        }
    },

    "modelSetting2": {
        "run_name": "xlmr_vit_cond_attn_multitask",
        "text_model": "xlmr",
        "image_model": "vit",
        "fusion_strategy": "conditional_attention",
        "loss": {
            "binary": {"type": "focal"},
            "multiclass": {"type": "weighted_ce", "class_weights": [1.0, 1.2, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]}, # Example weights for 8 classes
            "explanation": {"type": "weighted_bce", "pos_weights": [0.8] * 10},
            "weights": {
                "binary": 1.0,
                "multiclass": 1.0,
                "explanation": 1.0
            }
        },
        "training": {
            "learning_rate": 1e-5,
            "batch_size": 4,
            "epochs": 10,
            "early_stopping": 5,
            "mixed_precision": True,
            "checkpoint_dir": "checkpoints"
        }
    }
}

"""
INSTRUCTIONS FOR ADJUSTMENTS (For Grading / Experimentation):

Nhóm 1: Thay cách lấy vector đặc trưng
- Attention pooling (1 điểm): Kiểm tra file `src/models/fusion/attention.py` để xem cách `CLS` vector được thay thế/kết hợp với weights từ toàn bộ token hidden states. Trong cấu hình, chọn `fusion_strategy: "attention"`.
- Gated fusion (1 điểm): Kiểm tra file `src/models/fusion/gated.py`. Vector kết hợp bằng cổng học được. Trong cấu hình, chọn `fusion_strategy: "gated"`.
- Conditional attention (2 điểm): Kiểm tra file `src/models/fusion/conditional_attention.py`. Trọng số attention không cố định mà phụ thuộc vào điều kiện (vd aspect/keywords). Cấu hình: `fusion_strategy: "conditional_attention"`.

Nhóm 2: Thay hàm loss
- Gated loss (1 điểm): Để thực hiện, cần chỉnh sửa `src/losses/multitask_loss.py`. Tính mask dựa trên label của aspect. Nếu aspect không xuất hiện, set loss của aspect đó = 0.
- Weighted loss (1 điểm): Gán trọng số thấp cho nhãn có độ tin cậy thấp. Có thể thiết lập cấu hình loss `type: "weighted_ce"` hoặc `"weighted_bce"` trong config này, và thay đổi `class_weights` / `pos_weights` tương ứng.
- Focal loss (2 điểm): Tăng trọng số mẫu khó, giảm ảnh hưởng mẫu dễ. Được hỗ trợ bởi type `"focal"`. Cài đặt chi tiết nằm ở `src/losses/focal_loss.py`. Trong config, đặt `type: "focal"`.
"""

# Default config fallback
DEFAULT_CONFIG = TRAINING_CONFIGS["baseline"]
