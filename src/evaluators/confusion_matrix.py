from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_confusion_matrix(true_labels, predicted_labels, output_path: Path):
    """
    Tính toán và vẽ Confusion Matrix, sau đó lưu thành file ảnh.
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return cm.tolist()