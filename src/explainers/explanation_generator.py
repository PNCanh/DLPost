from explainers.lime_explainer import LimeMultimodalExplainer
from explainers.shap_explainer import ShapMultimodalExplainer

class ExplanationGenerator:
    """
    Factory class để khởi tạo explainer phù hợp (LIME hoặc SHAP).
    """
    @staticmethod
    def get_explainer(method: str, model, **kwargs):
        if method.lower() == "lime":
            return LimeMultimodalExplainer(model, **kwargs)
        elif method.lower() == "shap":
            return ShapMultimodalExplainer(model, **kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")