import torch
import torch.nn as nn

def get_layer_wise_optimizer_grouped_parameters(model, learning_rate, weight_decay=0.01, lr_decay=0.95):
    """
    Áp dụng Layer-wise Learning Rate (LLRD) cho text backbone (ví dụ PhoBERT, XLMR).
    Layers gần đầu vào có learning rate nhỏ hơn so với các layers gần output.
    """
    opt_parameters = []
    
    # Không áp dụng decay cho bias và LayerNorm
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    
    # Tìm kiếm các parameters của transformer layers
    # (Phần này có thể cần tùy chỉnh tùy theo cấu trúc model (ví dụ model.text_model.encoder.layer))
    try:
        layers = model.text_model.encoder.layer
        embeddings = model.text_model.embeddings
    except AttributeError:
        # Nếu model không phải dạng chuẩn (e.g., không dùng từ AutoModel.from_pretrained)
        return [{"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr": learning_rate}]

    num_layers = len(layers)
    
    # Head và các module không thuộc backbone lấy max learning rate
    head_params = []
    for name, param in model.named_parameters():
        if "text_model" not in name and param.requires_grad:
            head_params.append((name, param))
            
    opt_parameters.append({
        "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": learning_rate,
    })
    opt_parameters.append({
        "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": learning_rate,
    })
    
    # Các transformer layers (từ top xuống bottom)
    lr = learning_rate
    for i in range(num_layers - 1, -1, -1):
        layer_params = [(n, p) for n, p in layers[i].named_parameters() if p.requires_grad]
        opt_parameters.append({
            "params": [p for n, p in layer_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr,
        })
        opt_parameters.append({
            "params": [p for n, p in layer_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        })
        lr *= lr_decay
        
    # Embeddings (layer đầu tiên)
    embedding_params = [(n, p) for n, p in embeddings.named_parameters() if p.requires_grad]
    opt_parameters.append({
        "params": [p for n, p in embedding_params if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": lr,
    })
    opt_parameters.append({
        "params": [p for n, p in embedding_params if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": lr,
    })
    
    return opt_parameters
