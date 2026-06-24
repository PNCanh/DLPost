"""
Model Configurations

Định nghĩa cấu hình chi tiết cho các components của models, losses, optimizer và fusion.

=== POOLING STRATEGY (Nhóm 1 - Cách lấy vector đặc trưng) ===
- "cls": Dùng CLS token (mặc định, baseline)
- "attention_pooling": Attention-weighted sum toàn bộ token hidden states (1 điểm)
- "gated_cls": Kết hợp CLS + attention pooling qua learned gate (1 điểm)

=== FUSION STRATEGY ===
- "concat": Nối text + image features đơn giản
- "attention": Attention fusion giữa text và image
- "gated": Gated fusion giữa text và image
- "conditional_attention": Attention phụ thuộc aspect/keyword (2 điểm)

=== LOSS TYPE (Nhóm 2) ===
- "ce": CrossEntropyLoss
- "bce": BCEWithLogitsLoss
- "focal": FocalLoss multiclass (2 điểm)
- "binary_focal": FocalLoss binary
- "weighted_ce": CE với class weights thủ công
- "risk_weighted_ce": CE với weights tự tính từ risk_level (1 điểm)
- "weighted_bce": BCE với pos_weight/confidence
- "gated": true → GatedMultiTaskLoss, mask loss khi legitimate (1 điểm)
"""

# Cấu hình đầy đủ cho một pipeline multimodal
TRAINING_CONFIGS = {

    # ============================================================
    # BASELINE: CLS pooling + Attention fusion + CE/BCE loss
    # ============================================================
    "baseline": {
        "run_name": "baseline_cls_attention_ce",
        "text_model": "phobert",
        "image_model": "resnet",
        "pooling_strategy": "cls",
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

    # ============================================================
    # NHÓM 1: THAY CÁCH LẤY VECTOR ĐẶC TRƯNG
    # ============================================================

    # --- Attention Pooling (1 điểm) ---
    # Thay CLS vector bằng attention-weighted sum toàn bộ hidden states
    "attention_pooling": {
        "run_name": "attn_pool_attention_ce",
        "text_model": "phobert",
        "image_model": "resnet",
        "pooling_strategy": "attention_pooling",
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

    # --- Gated CLS + Attention Pooling (1 điểm) ---
    # Kết hợp CLS và attention pooling qua learned gate
    "gated_cls_pooling": {
        "run_name": "gated_cls_gated_ce",
        "text_model": "phobert",
        "image_model": "resnet",
        "pooling_strategy": "gated_cls",
        "fusion_strategy": "gated",
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

    # --- Conditional Attention (2 điểm) ---
    # Vector w phụ thuộc vào aspect (keyword_vector)
    "conditional_attention": {
        "run_name": "cls_cond_attn_ce",
        "text_model": "phobert",
        "image_model": "resnet",
        "pooling_strategy": "cls",
        "fusion_strategy": "conditional_attention",
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

    # ============================================================
    # NHÓM 2: THAY HÀM LOSS
    # ============================================================

    # --- Gated Loss (1 điểm) ---
    # Mask multiclass + explanation loss khi binary_label == 0 (legitimate)
    "gated_loss": {
        "run_name": "cls_attention_gated_loss",
        "text_model": "phobert",
        "image_model": "resnet",
        "pooling_strategy": "cls",
        "fusion_strategy": "attention",
        "loss": {
            "gated": True,
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

    # --- Weighted Loss (1 điểm) ---
    # Class weights dựa trên risk_level từ label metadata
    "weighted_loss": {
        "run_name": "cls_attention_weighted_loss",
        "text_model": "phobert",
        "image_model": "resnet",
        "pooling_strategy": "cls",
        "fusion_strategy": "attention",
        "loss": {
            "binary": {"type": "bce"},
            "multiclass": {
                "type": "risk_weighted_ce",
                "num_classes": 8
                # Weights tự tính từ risk_level:
                # legitimate(0)=0.3, job_scam(5)=1.0, investment_scam(5)=1.0,
                # sale_scam(4)=0.86, fake_image(3)=0.72, prize_scam(5)=1.0,
                # fake_course(4)=0.86, suspicious(2)=0.58
            },
            "explanation": {
                "type": "weighted_bce",
                "pos_weights": [1.0] * 10,
                "confidence_weights": [0.8] * 10
            },
            "weights": {
                "binary": 1.0,
                "multiclass": 1.5,
                "explanation": 1.0
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

    # --- Focal Loss (2 điểm) ---
    # Tăng trọng số mẫu khó, giảm ảnh hưởng mẫu dễ
    "focal_loss": {
        "run_name": "cls_attention_focal_loss",
        "text_model": "phobert",
        "image_model": "resnet",
        "pooling_strategy": "cls",
        "fusion_strategy": "attention",
        "loss": {
            "binary": {
                "type": "binary_focal",
                "alpha": 0.25,
                "gamma": 2.0
            },
            "multiclass": {
                "type": "focal",
                "alpha": 1.0,
                "gamma": 2.0
            },
            "explanation": {
                "type": "binary_focal",
                "alpha": 0.25,
                "gamma": 2.0
            },
            "weights": {
                "binary": 1.0,
                "multiclass": 1.0,
                "explanation": 0.5
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

    # ============================================================
    # COMBINED: Kết hợp tốt nhất (ví dụ)
    # ============================================================

    # ViBERT + ViT + Gated fusion + Focal loss
    "modelSetting1": {
        "run_name": "vibert_vit_gated_focal",
        "text_model": "vibert",
        "image_model": "vit",
        "pooling_strategy": "gated_cls",
        "fusion_strategy": "gated",
        "loss": {
            "binary": {"type": "binary_focal", "alpha": 0.25, "gamma": 2.0},
            "multiclass": {"type": "focal", "alpha": 1.0, "gamma": 2.0},
            "explanation": {"type": "weighted_bce", "pos_weights": [0.8] * 10},
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

    # XLM-R + ViT + Conditional Attention + Gated Loss + Weighted CE
    "modelSetting2": {
        "run_name": "xlmr_vit_cond_attn_gated_weighted",
        "text_model": "xlmr",
        "image_model": "vit",
        "pooling_strategy": "attention_pooling",
        "fusion_strategy": "conditional_attention",
        "loss": {
            "gated": True,
            "binary": {"type": "binary_focal", "alpha": 0.25, "gamma": 2.0},
            "multiclass": {
                "type": "risk_weighted_ce",
                "num_classes": 8
            },
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
- Attention pooling (1 điểm): Đặt pooling_strategy: "attention_pooling".
  Module: src/models/pooling/attention_pooling.py
  Tính attention weights trên toàn bộ token hidden states, dùng weighted sum thay CLS.

- Gated fusion (1 điểm): Đặt pooling_strategy: "gated_cls".
  Module: src/models/pooling/gated_pooling.py
  Kết hợp CLS + attention pooling qua gate: g*CLS + (1-g)*attn_pool.

- Conditional attention (2 điểm): Đặt fusion_strategy: "conditional_attention".
  Module: src/models/fusion/conditional_attention.py
  Vector w phụ thuộc vào keyword_vector (aspect), không cố định.

Nhóm 2: Thay hàm loss
- Gated loss (1 điểm): Đặt loss.gated: true.
  Module: src/losses/gated_loss.py
  Nếu binary_label == 0 → mask multiclass_loss = 0 và explanation_loss = 0.

- Weighted loss (1 điểm): Đặt multiclass loss type: "risk_weighted_ce".
  Module: src/losses/weighted_loss.py
  Tự tính class weights từ risk_level. Nhãn risk thấp → weight thấp.

- Focal loss (2 điểm): Đặt loss type: "focal" hoặc "binary_focal".
  Module: src/losses/focal_loss.py
  Tăng trọng số mẫu khó (1-pt)^gamma, giảm ảnh hưởng mẫu dễ.
"""

# Default config fallback
DEFAULT_CONFIG = TRAINING_CONFIGS["baseline"]
