from pathlib import Path

import pandas as pd

from configs import paths

from data.loaders.post_loader import (
    load_post_json
)

from data.loaders.image_loader import (
    discover_images
)


def build_image_dataset():

    records = []

    raw_dir = paths.RAW_DIR

    from data.builders.label_encoder import load_label_mapping
    label_mapping = load_label_mapping()

    for post_dir in sorted(
        [
            p
            for p in raw_dir.iterdir()
            if p.is_dir()
        ]
    ):

        post = load_post_json(
            post_dir
        )

        if post is None:

            continue

        images = discover_images(
            post_dir
        )

        for image_path in images:

            records.append(

                {

                    "post_id":
                        post[
                            "post_id"
                        ],

                    "image_path":
                        image_path,

                    "binary_label":
                        int(
                            post[
                                "binary_label"
                            ]
                        ),

                    "multi_label":
                        label_mapping.get(post.get("multi_label", "suspicious"), 7)
                }
            )

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
                    "image_path": image_path,
                    "binary_label": binary_label,
                    "multi_label": multi_label
                })

    df = pd.DataFrame(
        records
    )

    output_file = paths.IMAGE_DATASET_FILE

    df.to_parquet(

        output_file,

        index=False
    )

    print(
        f"\nSaved "
        f"{len(df)} images"
    )

    print(
        output_file
    )