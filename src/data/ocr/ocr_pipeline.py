import os
import easyocr
from pathlib import Path
from tqdm import tqdm
from configs import paths

class OCRPipeline:
    def __init__(self, languages=['vi', 'en'], gpu=True):
        print(f"[OCR] Khởi tạo EasyOCR (languages={languages}, gpu={gpu})...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
    def run_ocr_on_dataset(self, raw_dir=paths.RAW_DIR):
        """
        Chạy OCR trên tất cả các thư mục post trong dataset trước khi tiền xử lý văn bản.
        """
        print(f"[OCR] Bắt đầu trích xuất text từ thư mục: {raw_dir}")
        post_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        
        if not post_dirs:
            print("[OCR] Không tìm thấy thư mục post nào!")
            return
            
        print(f"[OCR] Tìm thấy {len(post_dirs)} posts.")
        
        extracted_count = 0
        for post_dir in tqdm(post_dirs, desc="[OCR] Progress"):
            ocr_text_lines = []
            
            # Tìm ảnh
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(list(post_dir.glob(ext)))
                
            for img_path in image_files:
                try:
                    results = self.reader.readtext(str(img_path))
                    for (_, text, _) in results:
                        ocr_text_lines.append(text)
                except Exception as e:
                    print(f"[OCR] Lỗi khi đọc {img_path}: {e}")
                    
            # Lưu ocr.txt
            ocr_out_path = post_dir / "ocr.txt"
            with open(ocr_out_path, "w", encoding="utf-8") as f:
                if ocr_text_lines:
                    f.write(" ".join(ocr_text_lines))
                    extracted_count += 1
                else:
                    f.write("")
                    
        print(f"[OCR] Hoàn thành. Đã trích xuất văn bản cho {extracted_count}/{len(post_dirs)} posts.")

if __name__ == "__main__":
    pipeline = OCRPipeline()
    pipeline.run_ocr_on_dataset()
