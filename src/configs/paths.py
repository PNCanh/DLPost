"""
Path Configuration

Tự động phát hiện môi trường (Local hoặc Google Colab) và thiết lập paths phù hợp.

- Local: Dùng đường dẫn tương đối từ project root
- Colab: Dùng Google Drive paths cho dataset/outputs (persistent storage)

Output structure:
├── outputs/
│   ├── checkpoints/        ← model weights (.pth, .h5)
│   ├── logs/               ← training logs
│   └── predictions/        ← kết quả dự đoán
│       ├── figures/        ← biểu đồ, charts
│       └── metrics/        ← file CSV kết quả
"""

import os
from pathlib import Path

# =========================
# Auto-detect Environment
# =========================

IS_COLAB = bool(os.environ.get("COLAB_RELEASE_TAG"))

if IS_COLAB:
    # Trên Colab: code ở /content/DLPost, data ở GDrive
    ROOT_DIR = Path("/content/DLPost")
    GDRIVE_ROOT = Path("/content/drive/MyDrive/DLPost")
    
    # Dataset và outputs nằm trên GDrive (persistent)
    DATASET_DIR = GDRIVE_ROOT / "dataset"
    OUTPUT_DIR = GDRIVE_ROOT / "outputs"
else:
    # Local: mọi thứ nằm trong project root
    ROOT_DIR = (
        Path(__file__)
        .resolve()
        .parents[2]
    )
    
    DATASET_DIR = ROOT_DIR / "dataset"
    OUTPUT_DIR = ROOT_DIR / "outputs"

# =========================
# Dataset
# =========================

RAW_DIR = (
    DATASET_DIR /
    "raw" /
    "posts"
)

RAW_IMAGES_DIR = (
    DATASET_DIR /
    "raw" /
    "images"
)

RESOURCE_DIR = (
    DATASET_DIR /
    "resources"
)

PROCESSED_DIR = (
    DATASET_DIR /
    "processed"
)

SPLIT_DIR = (
    DATASET_DIR /
    "splitted"
)

# =========================
# Output
# =========================

CHECKPOINTS_DIR = (
    OUTPUT_DIR /
    "checkpoints"
)

LOGS_DIR = (
    OUTPUT_DIR /
    "logs"
)

PREDICTIONS_DIR = (
    OUTPUT_DIR /
    "predictions"
)

FIGURES_DIR = (
   OUTPUT_DIR /
    "figures"
)

METRICS_DIR = (
    OUTPUT_DIR /
    "metrics"
)

# =========================
# Resource Files
# =========================

LABELS_FILE = (
    RESOURCE_DIR /
    "labels.json"
)

EXPLANATION_LABELS_FILE = (
    RESOURCE_DIR /
    "explanation_labels.json"
)

SCAM_KEYWORDS_FILE = (
    RESOURCE_DIR /
    "scam_keywords.json"
)

TEENCODE_FILE = (
    RESOURCE_DIR /
    "teencode.json"
)

ABBREVIATIONS_FILE = (
    RESOURCE_DIR /
    "abbreviations.json"
)

STOPWORDS_FILE = (
    RESOURCE_DIR /
    "stopwords.txt"
)

# =========================
# Templates
# =========================

TEMPLATES_DIR = (
    RESOURCE_DIR /
    "templates"
)

LEGITIMATE_TEMPLATE_DIR = (
    TEMPLATES_DIR /
    "legitimate"
)

JOB_SCAM_TEMPLATE_DIR = (
    TEMPLATES_DIR /
    "job_scam"
)

INVESTMENT_SCAM_TEMPLATE_DIR = (
    TEMPLATES_DIR /
    "invesment_scam"
)

SALE_SCAM_TEMPLATE_DIR = (
    TEMPLATES_DIR /
    "sale_scam"
)

PRIZE_SCAM_TEMPLATE_DIR = (
    TEMPLATES_DIR /
    "prize_scam"
)

FAKE_COURSE_TEMPLATE_DIR = (
    TEMPLATES_DIR /
    "fake_course"
)

SUSPICIOUS_TEMPLATE_DIR = (
    TEMPLATES_DIR /
    "suspicious"
)

# =========================
# Processed Dataset
# =========================

PROCESSED_DATASET_FILE = (
    PROCESSED_DIR /
    "processed_dataset.parquet"
)

IMAGE_DATASET_FILE = (
    PROCESSED_DIR /
    "image_dataset.parquet"
)

# =========================
# Split Dataset
# =========================

TRAIN_CSV_FILE = (
    SPLIT_DIR /
    "train.csv"
)

VAL_CSV_FILE = (
    SPLIT_DIR /
    "val.csv"
)

TEST_CSV_FILE = (
    SPLIT_DIR /
    "test.csv"
)

TRAIN_PARQUET_FILE = (
    SPLIT_DIR /
    "train.parquet"
)

VAL_PARQUET_FILE = (
    SPLIT_DIR /
    "val.parquet"
)

TEST_PARQUET_FILE = (
    SPLIT_DIR /
    "test.parquet"
)

IMAGE_TRAIN_PARQUET_FILE = (
    SPLIT_DIR /
    "image_train.parquet"
)

IMAGE_VAL_PARQUET_FILE = (
    SPLIT_DIR /
    "image_val.parquet"
)

IMAGE_TEST_PARQUET_FILE = (
    SPLIT_DIR /
    "image_test.parquet"
)


def ensure_directories():
    """
    Tạo tất cả thư mục cần thiết nếu chưa tồn tại.
    Gọi hàm này khi khởi tạo pipeline.
    """
    dirs = [
        DATASET_DIR,
        RAW_DIR,
        RAW_IMAGES_DIR,
        RESOURCE_DIR,
        PROCESSED_DIR,
        SPLIT_DIR,
        OUTPUT_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        PREDICTIONS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)