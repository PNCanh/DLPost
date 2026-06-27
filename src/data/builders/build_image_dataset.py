from pathlib import Path
import pandas as pd
from configs import paths
from data.loaders.post_loader import load_post_json
from data.loaders.image_loader import discover_images

def build_image_dataset(kaggle_dir=None):
    records = []
    raw_dir = paths.RAW_DIR

    from data.builders.label_encoder import load_label_mapping
    label_mapping = load_label_mapping()

    # 1. Quét ảnh từ các post
    for post_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        post = load_post_json(post_dir)
        if post is None:
            continue
        images = discover_images(post_dir)
        for image_path in images:
            records.append({
                "post_id": post["post_id"],
                "image_path": str(image_path),
                "binary_label": int(post["binary_label"]),
                "multi_label": label_mapping.get(post.get("multi_label", "suspicious"), 7)
            })

    # 2. Quét ảnh standalone từ raw_images_dir
    raw_images_dir = paths.RAW_IMAGES_DIR
    if raw_images_dir.exists():
        for label_folder in ["fake_image", "legitimate"]:
            folder_path = raw_images_dir / label_folder
            if not folder_path.exists() or not folder_path.is_dir():
                continue
                
            images = discover_images(folder_path)
            binary_label = 1 if label_folder != "legitimate" else 0
            multi_label = label_mapping.get(label_folder, 7)
            
            for image_path in images:
                records.append({
                    "post_id": "",  # Empty for standalone images
                    "image_path": str(image_path),
                    "binary_label": binary_label,
                    "multi_label": multi_label
                })

    # 3. Quét ảnh từ Kaggle Dataset nếu đường dẫn được truyền vào
    if kaggle_dir:
        kaggle_path = Path(kaggle_dir)
        if kaggle_path.exists():
            print(f"[KAGGLE] Đang quét dataset Kaggle từ đường dẫn: {kaggle_path}")
            
            # Quét đệ quy tất cả ảnh
            kaggle_images = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                kaggle_images.extend(list(kaggle_path.rglob(ext)))
                
            print(f"[KAGGLE] Tìm thấy {len(kaggle_images)} ảnh.")
            
            for img_path in kaggle_images:
                path_lower = str(img_path).lower()
                
                # Phân loại dựa trên tên thư mục chứa ảnh
                if 'ai' in path_lower or 'fake' in path_lower or 'generated' in path_lower:
                    binary_label = 1
                    multi_label = 4 # fake_image
                elif 'real' in path_lower or 'natural' in path_lower or 'legit' in path_lower:
                    binary_label = 0
                    multi_label = 0 # legitimate
                else:
                    binary_label = 0
                    multi_label = 7 # suspicious
                    
                records.append({
                    "post_id": "",  # Ảnh từ kaggle không gắn liền với post_id cụ thể
                    "image_path": str(img_path),
                    "binary_label": binary_label,
                    "multi_label": multi_label
                })
        else:
            print(f"[KAGGLE ERROR] Đường dẫn Kaggle không tồn tại: {kaggle_path}")

    df = pd.DataFrame(records)
    output_file = paths.IMAGE_DATASET_FILE
    df.to_parquet(output_file, index=False)
    
    print(f"\n[IMAGE BUILD] Đã lưu {len(df)} images vào {output_file}")
    return df

if __name__ == "__main__":
    build_image_dataset()