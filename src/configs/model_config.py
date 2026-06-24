"""
Model Configurations

"""

# Cấu hình đầy đủ cho một pipeline multimodal
TRAINING_CONFIGS = {
    "baseline": {
    "run_name": "baseline_phobert_resnet",

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
        "batch_size": 16,
        "epochs": 15,
        "early_stopping": 5,
        "mixed_precision": True,
        "checkpoint_dir": "checkpoints",
        "scheduler": "cosine",
        "warmup_ratio": 0.1
    }
},

"production_candidate": {
    "run_name": "phobert_resnet_gated_focal",

    "text_model": "phobert",
    "image_model": "resnet",

    "pooling_strategy": "gated_cls",
    "fusion_strategy": "gated",

    "loss": {
        "gated": True,

        "binary": {
            "type": "binary_focal",
            "alpha": 0.5,
            "gamma": 2.0
        },

        "multiclass": {
            "type": "focal",
            "alpha": 1.0,
            "gamma": 2.0
        },

        "explanation": {
            "type": "binary_focal",
            "alpha": 0.5,
            "gamma": 2.0
        },

        "weights": {
            "binary": 1.0,
            "multiclass": 1.5,
            "explanation": 1.0
        }
    },

    "training": {
        "learning_rate": 1e-5,
        "batch_size": 16,
        "epochs": 15,
        "early_stopping": 5,
        "mixed_precision": True,
        "checkpoint_dir": "checkpoints",
        "scheduler": "cosine",
        "warmup_ratio": 0.1
    }
},

"best_experiment": {
    "run_name": "xlmr_conditional_weighted",

    "text_model": "xlmr",
    "image_model": "vit",

    "pooling_strategy": "attention_pooling",
    "fusion_strategy": "conditional_attention",

    "loss": {
        "gated": True,

        "binary": {
            "type": "binary_focal",
            "alpha": 0.5,
            "gamma": 2.0
        },

        "multiclass": {
            "type": "risk_weighted_ce",
            "num_classes": 8
        },

        "explanation": {
            "type": "weighted_bce",
            "pos_weights": [1.0] * 10
        },

        "weights": {
            "binary": 1.0,
            "multiclass": 1.5,
            "explanation": 1.0
        }
    },

    "training": {
        "learning_rate": 1e-5,
        "batch_size": 8,
        "epochs": 20,
        "early_stopping": 7,
        "mixed_precision": True,
        "checkpoint_dir": "checkpoints",
        "scheduler": "cosine",
        "warmup_ratio": 0.1
    }
}
}

DEFAULT_CONFIGS = [
    "baseline",
    "production_candidate",
    "best_experiment"
]
