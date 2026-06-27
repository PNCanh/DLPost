import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAttention(nn.Module):
    """
    Task-specific Attention Module for Parallel Multi-task Learning.
    Học cách tập trung vào những features quan trọng cho một task cụ thể.
    """
    def __init__(self, hidden_dim: int, num_tasks: int = 3):
        super(TaskAttention, self).__init__()
        self.num_tasks = num_tasks
        # Mỗi task sẽ có một weight vector riêng để compute attention scores
        self.task_query = nn.Parameter(torch.Tensor(num_tasks, hidden_dim))
        nn.init.normal_(self.task_query, mean=0.0, std=0.02)
        
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, features: torch.Tensor):
        """
        features: shape [batch_size, seq_len, hidden_dim] hoặc [batch_size, hidden_dim]
        Nếu là [batch_size, hidden_dim], unsqueeze thành [batch_size, 1, hidden_dim]
        """
        if features.dim() == 2:
            features = features.unsqueeze(1) # [batch, 1, hidden]
            
        batch_size = features.size(0)
        
        keys = self.key_layer(features)   # [batch, seq_len, hidden]
        values = self.value_layer(features) # [batch, seq_len, hidden]
        
        # task_query: [num_tasks, hidden]
        # keys: [batch, seq_len, hidden]
        # attention scores: [batch, num_tasks, seq_len]
        # = keys (batch, seq, hidden) * query^T (hidden, num_tasks)
        query = self.task_query.unsqueeze(0).expand(batch_size, -1, -1) # [batch, num_tasks, hidden]
        
        scores = torch.bmm(query, keys.transpose(1, 2)) * self.scale # [batch, num_tasks, seq_len]
        attn_weights = F.softmax(scores, dim=-1) # [batch, num_tasks, seq_len]
        
        # Áp dụng attention weights lên values
        # [batch, num_tasks, seq_len] x [batch, seq_len, hidden] -> [batch, num_tasks, hidden]
        task_features = torch.bmm(attn_weights, values)
        
        # Trả về list các feature vectors cho từng task
        return [task_features[:, i, :] for i in range(self.num_tasks)]
