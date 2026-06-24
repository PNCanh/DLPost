"""
Keyword Encoder

Chuyển đổi nội dung text thành keyword vector dựa trên scam_keywords.json.

Keyword vector là một binary/count vector đại diện cho sự xuất hiện
của các scam keywords trong bài đăng. Vector này được dùng làm:
- Input cho Conditional Attention (aspect-dependent attention)
- Feature bổ sung trong fusion layer

Cấu trúc scam_keywords.json:
{
    "category_name": {
        "with_value": {
            "keyword": ["value1", "value2", ...]
        },
        "without_value": ["keyword1", "keyword2", ...]
    }
}
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch


class KeywordEncoder:
    """
    Encode text thành keyword vector.
    
    Mỗi keyword/pattern trở thành một dimension trong vector.
    Giá trị = 1 nếu keyword xuất hiện trong text, 0 nếu không.
    
    Args:
        keywords_path: Đường dẫn đến file scam_keywords.json
    """
    
    def __init__(self, keywords_path: str = None):
        if keywords_path is None:
            from configs.paths import SCAM_KEYWORDS_FILE
            keywords_path = SCAM_KEYWORDS_FILE
            
        self.keywords_path = Path(keywords_path)
        self.keyword_list = []  # Danh sách tất cả keywords (flat)
        self.keyword_categories = {}  # Mapping keyword -> category
        self.category_names = []  # Danh sách category names
        
        self._load_keywords()
    
    def _load_keywords(self):
        """Load và flatten keywords từ JSON file."""
        if not self.keywords_path.exists():
            print(f"⚠️  Keywords file not found: {self.keywords_path}")
            return
            
        with open(self.keywords_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.category_names = list(data.keys())
        
        for category, content in data.items():
            # Without value keywords (simple patterns)
            without_value = content.get("without_value", [])
            for keyword in without_value:
                self.keyword_list.append(keyword.lower())
                self.keyword_categories[keyword.lower()] = category
            
            # With value keywords (keyword + value combinations)
            with_value = content.get("with_value", {})
            for keyword, values in with_value.items():
                # Thêm keyword gốc
                self.keyword_list.append(keyword.lower())
                self.keyword_categories[keyword.lower()] = category
                
                # Thêm keyword + value combinations
                for value in values:
                    combined = f"{keyword} {value}".lower()
                    self.keyword_list.append(combined)
                    self.keyword_categories[combined] = category
    
    @property
    def vector_dim(self):
        """Kích thước của keyword vector."""
        return len(self.keyword_list)
    
    def encode(self, text: str) -> List[float]:
        """
        Encode một text thành keyword vector.
        
        Args:
            text: Nội dung bài đăng
            
        Returns:
            List[float]: Binary vector, 1 nếu keyword xuất hiện
        """
        text_lower = text.lower()
        vector = []
        
        for keyword in self.keyword_list:
            if keyword in text_lower:
                vector.append(1.0)
            else:
                vector.append(0.0)
        
        return vector
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode một batch texts thành keyword vectors.
        
        Args:
            texts: Danh sách nội dung bài đăng
            
        Returns:
            torch.Tensor: (batch_size, vector_dim)
        """
        vectors = [self.encode(text) for text in texts]
        return torch.tensor(vectors, dtype=torch.float32)
    
    def get_matched_keywords(self, text: str) -> Dict[str, List[str]]:
        """
        Trả về danh sách keywords matched, nhóm theo category.
        Hữu ích cho việc giải thích kết quả.
        
        Args:
            text: Nội dung bài đăng
            
        Returns:
            Dict[str, List[str]]: {category: [matched_keywords]}
        """
        text_lower = text.lower()
        matched = {}
        
        for keyword in self.keyword_list:
            if keyword in text_lower:
                category = self.keyword_categories[keyword]
                if category not in matched:
                    matched[category] = []
                matched[category].append(keyword)
        
        return matched
    
    def get_category_vector(self, text: str) -> List[float]:
        """
        Encode text thành category-level vector (1 dimension per category).
        Useful khi muốn vector nhỏ hơn.
        
        Args:
            text: Nội dung bài đăng
            
        Returns:
            List[float]: Vector có kích thước = số categories
        """
        text_lower = text.lower()
        category_scores = {cat: 0.0 for cat in self.category_names}
        
        for keyword in self.keyword_list:
            if keyword in text_lower:
                category = self.keyword_categories[keyword]
                category_scores[category] += 1.0
        
        return [category_scores[cat] for cat in self.category_names]
