from pathlib import Path

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

# =========================
# Dataset
# =========================

DATASET_DIR = (
    ROOT_DIR /
    "dataset"
)

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
    "split"
)

# =========================
# Output
# =========================

OUTPUT_DIR = (
    ROOT_DIR /
    "outputs"
)

CHECKPOINTS_DIR = (
    OUTPUT_DIR /
    "checkpoints"
)

CONFUSION_MATRIX_DIR = (
    OUTPUT_DIR /
    "confusion_matrix"
)

EXPLANATIONS_DIR = (
    OUTPUT_DIR /
    "explanations"
)

REPORTS_DIR = (
    OUTPUT_DIR /
    "reports"
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