from sklearn.metrics import classification_report

def generate_classification_report(true_labels, predicted_labels):
    """
    Tạo classification report chi tiết (dạng dict).
    """
    report_dict = classification_report(
        true_labels, 
        predicted_labels, 
        output_dict=True,
        zero_division=0
    )
    return report_dict