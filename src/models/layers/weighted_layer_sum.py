import torch
import torch.nn as nn

class WeightedLayerSum(nn.Module):
    """
    Weighted Layer Sum cho các hidden states của Transformer (ví dụ: RoBERTa, BERT).
    Cho phép mô hình tự học trọng số của từng layer khi kết hợp.
    """
    def __init__(self, num_layers: int = 12):
        super(WeightedLayerSum, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, all_hidden_states):
        """
        all_hidden_states: Tuple of hidden states từ Transformer.
        Thường thì layer đầu tiên là embeddings, chúng ta lấy từ layer 1 đến num_layers.
        Shape của mỗi hidden_state: [batch_size, seq_len, hidden_dim]
        """
        # Stack theo chiều cuối -> [batch, seq_len, hidden, num_layers]
        # all_hidden_states có thể có 13 phần tử (1 embedding + 12 layers)
        if len(all_hidden_states) > len(self.weights):
            hidden_states = torch.stack(all_hidden_states[1:], dim=-1)
        else:
            hidden_states = torch.stack(all_hidden_states, dim=-1)
            
        # Áp dụng softmax để trọng số cộng lại bằng 1
        norm_weights = self.softmax(self.weights)
        
        # Nhân weights và cộng lại
        # shape: [batch, seq_len, hidden]
        weighted_sum = (hidden_states * norm_weights).sum(-1)
        return weighted_sum
