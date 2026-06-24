"""
Classification Evaluator

Tính toán metrics, confusion matrix, classification report và lưu kết quả.

Tên file output theo format: {model_name}_{MMHHddmmyy}_{loại_file}
Ví dụ:
- phobert_resnet_attention_0214250625_report.json      → predictions/metrics/
- phobert_resnet_attention_0214250625_cm.png            → predictions/figures/
- phobert_resnet_attention_0214250625_metrics.csv       → predictions/metrics/
- phobert_resnet_attention_0214250625_predictions.json  → predictions/
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from configs.paths import PREDICTIONS_DIR, FIGURES_DIR, METRICS_DIR
from evaluators.metrics import compute_metrics
from evaluators.classification_report import generate_classification_report
from evaluators.confusion_matrix import generate_confusion_matrix

class ClassificationEvaluator:
    """
    Evaluator cho classification model.
    
    Lưu kết quả vào cấu trúc:
    outputs/predictions/
    ├── {model}_{timestamp}_predictions.json
    ├── figures/
    │   └── {model}_{timestamp}_cm.png
    └── metrics/
        ├── {model}_{timestamp}_report.json
        └── {model}_{timestamp}_metrics.csv
    """
    
    def __init__(self, run_name: str, model_name: str, timestamp: str = None):
        self.run_name = run_name
        self.model_name = model_name
        # Format: MMHHddmmyy
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%M%H%d%m%y")
        
        # File prefix: model_name + timestamp
        self.file_prefix = f"{self.model_name}_{self.timestamp}"
        
        # Tạo thư mục nếu chưa có
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, predictions: list, metrics: dict):
        """
        Calculate classification metrics, confusion matrix, and save them.
        
        Output files:
        - predictions/{prefix}_predictions.json
        - predictions/figures/{prefix}_cm.png
        - predictions/metrics/{prefix}_report.json
        - predictions/metrics/{prefix}_metrics.csv
        """
        if not predictions:
            print("No predictions to evaluate.")
            return
            
        true_labels = [p["true_label"] for p in predictions]
        predicted_labels = [p["predicted_label"] for p in predictions]
        
        # 1. Compute basic metrics
        basic_metrics = compute_metrics(true_labels, predicted_labels)
        
        # 2. Compute classification report
        class_report = generate_classification_report(true_labels, predicted_labels)
        
        # 3. Generate confusion matrix plot → figures/
        cm_filename = f"{self.file_prefix}_cm.png"
        cm_path = FIGURES_DIR / cm_filename
        cm_data = generate_confusion_matrix(true_labels, predicted_labels, cm_path)
        
        # 4. Save full report → metrics/
        report = {
            "model_name": self.model_name,
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "training_metrics": metrics,
            "evaluation_metrics": basic_metrics,
            "classification_report": class_report,
            "confusion_matrix": cm_data
        }
        
        report_filename = f"{self.file_prefix}_report.json"
        with open(METRICS_DIR / report_filename, "w", encoding="utf8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        # 5. Save metrics CSV → metrics/
        csv_filename = f"{self.file_prefix}_metrics.csv"
        with open(METRICS_DIR / csv_filename, "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in basic_metrics.items():
                writer.writerow([k, f"{v:.4f}"])
            
        # 6. Save predictions → predictions/
        pred_filename = f"{self.file_prefix}_predictions.json"
        serializable_preds = []
        for p in predictions:
            pred_copy = dict(p)
            # Convert numpy arrays to lists
            for key in ["explanation_probs", "multiclass_probs", "binary_probs"]:
                if key in pred_copy and hasattr(pred_copy[key], "tolist"):
                    pred_copy[key] = pred_copy[key].tolist()
            serializable_preds.append(pred_copy)
            
        with open(PREDICTIONS_DIR / pred_filename, "w", encoding="utf8") as f:
            json.dump(serializable_preds, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Evaluation saved:")
        print(f"   Predictions: {PREDICTIONS_DIR / pred_filename}")
        print(f"   CM Figure:   {cm_path}")
        print(f"   Report:      {METRICS_DIR / report_filename}")
        print(f"   Metrics CSV: {METRICS_DIR / csv_filename}")
