import json
import re
from pathlib import Path
from configs import paths

class TextPreprocessor:
    def __init__(self, teencode_file=None, abbreviations_file=None):
        # Tải các từ điển
        self.keywords = self._load_dict(paths.KEYWORDS_FILE)
        self.teencode = self._load_dict(teencode_file or paths.TEENCODE_FILE)
        self.abbreviations = self._load_dict(abbreviations_file or paths.ABBREVIATIONS_FILE)
        self.stopwords = self._load_dict(paths.STOPWORDS_FILE)
        
        # Stopwords thường là list, nếu là dict thì lấy keys
        if isinstance(self.stopwords, dict):
            self.stopwords = set(self.stopwords.keys())
        elif isinstance(self.stopwords, list):
            self.stopwords = set(self.stopwords)
        else:
            self.stopwords = set()
            
    def _load_dict(self, filepath: Path) -> dict:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
            
        # 1. Chuyển về chữ thường
        text = text.lower()
        
        # 2. Xóa dấu xuống dòng
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # 3. Tách/chuyển ngữ nghĩa icon thành văn bản (giả sử có trong abbreviations/teencode)
        # 4. Chuyển hóa teencode/viết tắt thành văn bản
        words = text.split()
        normalized_words = []
        for word in words:
            if word in self.abbreviations:
                normalized_words.extend(self.abbreviations[word].split())
            elif word in self.teencode:
                normalized_words.extend(self.teencode[word].split())
            else:
                normalized_words.append(word)
                
        # 5. Xóa dấu câu trong đoạn văn
        text = " ".join(normalized_words)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 6. Xóa stop word
        words = text.split()
        final_words = [w for w in words if w not in self.stopwords]
        
        return " ".join(final_words).strip()