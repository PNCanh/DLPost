import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from configs.paths import REPORTS_DIR

def generate_comparison_report():
    print("\n[Generating Comparison Report]")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read all json files in REPORTS_DIR
    report_files = list(REPORTS_DIR.glob("*_report.json"))
    if not report_files:
        print("No report files found in", REPORTS_DIR)
        return

    data = []
    for file_path in report_files:
        try:
            with open(file_path, "r", encoding="utf8") as f:
                report = json.load(f)
                
            model_name = report.get("model_name", "Unknown")
            # Parse classification report
            cls_rep = report.get("classification_report", {})
            eval_metrics = report.get("evaluation_metrics", {})
            
            # Extract metrics
            # 0: legitimate (negative), 1: scam (positive)
            # Default to 0 if not found
            acc = eval_metrics.get("accuracy", cls_rep.get("accuracy", 0))
            
            neg_stats = cls_rep.get("0", cls_rep.get("0.0", {}))
            pos_stats = cls_rep.get("1", cls_rep.get("1.0", {}))
            macro_stats = cls_rep.get("macro avg", {})
            
            precision_neg = neg_stats.get("precision", 0)
            recall_neg = neg_stats.get("recall", 0)
            f1_neg = neg_stats.get("f1-score", 0)
            
            precision_pos = pos_stats.get("precision", 0)
            recall_pos = pos_stats.get("recall", 0)
            f1_pos = pos_stats.get("f1-score", 0)
            
            macro_f1 = macro_stats.get("f1-score", 0)
            
            data.append({
                "Model": model_name,
                "Accuracy": acc,
                "Precision_negative": precision_neg,
                "Recall_negative": recall_neg,
                "F1_negative": f1_neg,
                "Precision_positive": precision_pos,
                "Recall_positive": recall_pos,
                "F1_positive": f1_pos,
                "Macro_F1": macro_f1
            })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if not data:
        print("No valid data extracted from reports.")
        return
        
    df = pd.DataFrame(data)
    
    # 1. Save table to CSV
    csv_path = REPORTS_DIR / "model_comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")
    
    # 2. Draw comparison charts based on: Accuracy, Recall_negative, F1_negative, Macro_F1
    chart_metrics = ["Accuracy", "Recall_negative", "F1_negative", "Macro_F1"]
    
    # Melt the dataframe for seaborn barplot
    df_melted = df.melt(id_vars=["Model"], value_vars=chart_metrics, 
                        var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model")
    plt.title("Model Comparison")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    chart_path = REPORTS_DIR / "model_comparison_chart.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved comparison chart to {chart_path}")
    
if __name__ == "__main__":
    generate_comparison_report()
