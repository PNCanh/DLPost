import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from configs import paths
import os

class MultimodalDataset(Dataset):
    """
    Dataset class cho dữ liệu Multimodal (Văn bản + Hình ảnh).
    Thực hiện tokenize văn bản và transform hình ảnh.
    """
    def __init__(self, df, text_col='text_clean', image_col='primary_image', 
                 tokenizer_name='vinai/phobert-base', max_len=256, transform=None):
        self.df = df
        self.text_col = text_col
        self.image_col = image_col
        
        # Text processing
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        
        # Image processing
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Xử lý Text
        text = str(row[self.text_col]) if pd.notna(row[self.text_col]) else ""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. Xử lý Image
        image_path = str(row[self.image_col]) if pd.notna(row[self.image_col]) else ""
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            except Exception:
                # Ảnh lỗi -> Tạo tensor 0
                image = torch.zeros(3, 224, 224)
        else:
            # Không có ảnh -> Tạo tensor 0
            image = torch.zeros(3, 224, 224)
            
        # 3. Lấy Labels
        binary_label = torch.tensor(row['binary_label'], dtype=torch.float32)
        multi_label = torch.tensor(row['multi_label'], dtype=torch.long)
        
        # Xử lý explanation_vector nếu có
        explanation_vector = torch.tensor(row.get('explanation_vector', [0]*10), dtype=torch.float32)
        
        # Xử lý keyword_vector
        keyword_vector = torch.tensor(row.get('keyword_vector', [0]*44), dtype=torch.float32)

        return {
            "input_ids": encoding['input_ids'].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "image": image,
            "binary_label": binary_label,
            "multi_label": multi_label,
            "explanation_vector": explanation_vector,
            "keyword_vector": keyword_vector,
            "post_id": row.get('post_id', str(idx))
        }

def get_dataloaders(config, train_df, val_df, test_df):
    """
    Helper function để tạo DataLoaders dựa trên model config.
    """
    tokenizer_map = {
        "phobert": "vinai/phobert-base",
        "vibert": "FPTAI/vibert-base-cased",
        "xlmr": "xlm-roberta-base"
    }
    tokenizer_name = tokenizer_map.get(config.get("text_model", "phobert"), "vinai/phobert-base")
    batch_size = config.get("training", {}).get("batch_size", 16)
    
    train_dataset = MultimodalDataset(train_df, tokenizer_name=tokenizer_name)
    val_dataset = MultimodalDataset(val_df, tokenizer_name=tokenizer_name)
    test_dataset = MultimodalDataset(test_df, tokenizer_name=tokenizer_name)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

class ImageOnlyDataset(Dataset):
    """
    Dataset class cho dữ liệu chỉ chứa Hình ảnh.
    """
    def __init__(self, df, image_col='image_path', transform=None):
        self.df = df
        self.image_col = image_col
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = str(row[self.image_col]) if pd.notna(row[self.image_col]) else ""
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            except Exception:
                image = torch.zeros(3, 224, 224)
        else:
            image = torch.zeros(3, 224, 224)
            
        binary_label = torch.tensor(row['binary_label'], dtype=torch.float32)
        multi_label = torch.tensor(row['multi_label'], dtype=torch.long)
        explanation_vector = torch.tensor([0.0] * 10, dtype=torch.float32)
        
        return {
            "image": image,
            "binary_label": binary_label,
            "multi_label": multi_label,
            "explanation_vector": explanation_vector,
            "post_id": row.get('post_id', ""),
            "image_path": image_path
        }

def get_image_dataloaders(config, train_df, val_df, test_df):
    """
    Helper function để tạo DataLoaders cho ImageOnlyDataset.
    """
    batch_size = config.get("training", {}).get("batch_size", 16)
    
    train_dataset = ImageOnlyDataset(train_df)
    val_dataset = ImageOnlyDataset(val_df)
    test_dataset = ImageOnlyDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
