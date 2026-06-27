import torch

class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training.
    Giúp tăng tính cường tráng (robustness) của mô hình chống lại adversarial attacks.
    """
    def __init__(self, model, epsilon=1.0, emb_name='word_embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        """
        Thêm nhiễu vào word embeddings dựa trên gradient.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """
        Khôi phục lại word embeddings ban đầu sau khi tính toán.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
# AWP và FGD có thể được implement tương tự theo bài báo gốc.
