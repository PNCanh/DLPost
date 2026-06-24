import json
from pathlib import Path
from datetime import datetime
from configs.paths import CONFUSION_MATRIX_DIR, REPORTS_DIR, EXPLANATIONS_DIR
from evaluators.metrics import compute_metrics
from evaluators.classification_report import generate_classification_report
from evaluators.confusion_matrix import generate_confusion_matrix

class ClassificationEvaluator:
    def __init__(self, run_name: str, model_name: str, timestamp: str = None):
        self.run_name = run_name
        self.model_name = model_name
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%M%H%m%d%y")
        
        self.cm_dir = CONFUSION_MATRIX_DIR
        self.cm_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.expl_dir = EXPLANATIONS_DIR / f"{self.model_name}_{self.timestamp}"
        self.expl_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, predictions: list, metrics: dict):
        """
        Calculate classification metrics, confusion matrix, and save them.
        """
        if not predictions:
            print("No predictions to evaluate.")
            return
            
        true_labels = [p["true_label"] for p in predictions]
        predicted_labels = [p["predicted_label"] for p in predictions]
        
        timestamp = self.timestamp
        
        # 1. Compute basic metrics
        basic_metrics = compute_metrics(true_labels, predicted_labels)
        
        # 2. Compute classification report
        class_report = generate_classification_report(true_labels, predicted_labels)
        
        # 3. Generate confusion matrix and plot
        cm_filename = f"{self.model_name}_{timestamp}_cm.png"
        cm_path = self.cm_dir / cm_filename
        cm_data = generate_confusion_matrix(true_labels, predicted_labels, cm_path)
        
        # 4. Save report
        report = {
            "model_name": self.model_name,
            "run_name": self.run_name,
            "timestamp": timestamp,
            "training_metrics": metrics,
            "evaluation_metrics": basic_metrics,
            "classification_report": class_report,
            "confusion_matrix": cm_data
        }
        
        report_filename = f"{self.model_name}_{timestamp}_report.json"
        with open(self.reports_dir / report_filename, "w", encoding="utf8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
            
        # 5. Save explanation result for each post
        for p in predictions:
            post_id = p.get("post_id", "unknown")
            is_scam = "scam" if p.get("is_scam", False) else "legitimate"
            reason = p.get("explanation_probs", [])
            
            if hasattr(reason, "tolist"):
                reason = reason.tolist()
            elif isinstance(reason, list):
                reason = [float(r) for r in reason]
                
            expl_data = {
                "post_id": post_id,
                "prediction": is_scam,
                "reason": reason
            }
            expl_filename = f"{post_id}.json"
            with open(self.expl_dir / expl_filename, "w", encoding="utf8") as f:
                json.dump(expl_data, f, ensure_ascii=False, indent=4)
            
        print(f"Evaluation report saved to {self.reports_dir}, CM to {self.cm_dir}, explanations to {self.expl_dir}")
