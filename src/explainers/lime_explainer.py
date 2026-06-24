import numpy as np
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer

class LimeMultimodalExplainer:
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names
        
        # Khởi tạo 2 explainer độc lập
        self.text_explainer = LimeTextExplainer(class_names=self.class_names)
        self.image_explainer = LimeImageExplainer()
        
    def explain_text(self, text: str, predict_fn, num_features=10):
        """
        Giải thích độc lập phần Text. predict_fn nhận vào mảng string và trả về mảng xác suất.
        """
        exp = self.text_explainer.explain_instance(
            text, 
            predict_fn, 
            num_features=num_features
        )
        return exp

    def explain_image(self, image: np.ndarray, predict_fn, num_features=10):
        """
        Giải thích độc lập phần Image. predict_fn nhận vào mảng ảnh và trả về mảng xác suất.
        """
        exp = self.image_explainer.explain_instance(
            image, 
            predict_fn, 
            top_labels=1, 
            hide_color=0, 
            num_samples=100
        )
        return exp