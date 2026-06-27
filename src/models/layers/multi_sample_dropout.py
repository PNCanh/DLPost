import torch
import torch.nn as nn

class MultiSampleDropout(nn.Module):
    """
    Multi-sample Dropout to prevent overfitting and improve generalization.
    Uses n different dropout masks and averages the outputs.
    """
    def __init__(self, p: float = 0.1, n_samples: int = 5):
        super(MultiSampleDropout, self).__init__()
        self.p = p
        self.n_samples = n_samples
        # Chúng ta có thể dùng nhiều module nn.Dropout với các mask khác nhau tự động
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(n_samples)])

    def forward(self, x: torch.Tensor, classifier: nn.Module) -> torch.Tensor:
        """
        x: features
        classifier: lớp phân loại cuối cùng (nn.Linear)
        """
        # Áp dụng từng dropout mask, cho qua classifier và lấy trung bình
        outputs = []
        for dropout in self.dropouts:
            h = dropout(x)
            outputs.append(classifier(h))
            
        # Tính trung bình theo chiều 0 (nếu outputs là list các tensor)
        mean_out = torch.mean(torch.stack(outputs, dim=0), dim=0)
        return mean_out
