from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from configs import paths


def split_dataset():

    input_file = paths.PROCESSED_DATASET_FILE

    output_dir = paths.SPLIT_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_parquet(
        input_file
    )

    
    image_df = pd.read_parquet(
        paths.IMAGE_DATASET_FILE
    )


    if len(df) == 0:

        raise ValueError(
            "Dataset is empty."
        )

    print(
        f"\nTotal samples: "
        f"{len(df)}"
    )

    print(
        "\nClass Distribution"
    )

    print(
        df["multi_label"]
        .value_counts()
        .sort_index()
    )

    label_counts = (
        df["multi_label"]
        .value_counts()
    )

    rare_labels = (
        label_counts[
            label_counts < 3
        ]
    )

    if len(rare_labels) > 0:

        print(
            "\n[WARNING]"
            " Rare classes found:"
        )

        print(
            rare_labels
        )

    train_df, temp_df = train_test_split(

        df,

        test_size=0.30,

        random_state=42,

        stratify=df[
            "multi_label"
        ]
    )

    val_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        random_state=42,

        stratify=temp_df[
            "multi_label"
        ]
    )
    
    train_post_ids = set(
    train_df["post_id"]
)

    val_post_ids = set(
        val_df["post_id"]
    )

    test_post_ids = set(
        test_df["post_id"]
    )

    # Lọc ảnh từ bài viết (có post_id)
    post_images_df = image_df[image_df["post_id"] != ""]
    # Lọc ảnh độc lập (không có post_id)
    standalone_images_df = image_df[image_df["post_id"] == ""]

    # Gán ảnh từ bài viết vào các tập tương ứng với bài viết
    image_train_df = post_images_df[
        post_images_df["post_id"]
        .isin(train_post_ids)
    ]

    image_val_df = post_images_df[
        post_images_df["post_id"]
        .isin(val_post_ids)
    ]

    image_test_df = post_images_df[
        post_images_df["post_id"]
        .isin(test_post_ids)
    ]

    # Split ảnh độc lập
    if not standalone_images_df.empty:
        sa_train, sa_temp = train_test_split(
            standalone_images_df,
            test_size=0.30,
            random_state=42,
            stratify=standalone_images_df["multi_label"]
        )
        sa_val, sa_test = train_test_split(
            sa_temp,
            test_size=0.50,
            random_state=42,
            stratify=sa_temp["multi_label"]
        )
        
        # Gộp lại
        image_train_df = pd.concat([image_train_df, sa_train], ignore_index=True)
        image_val_df = pd.concat([image_val_df, sa_val], ignore_index=True)
        image_test_df = pd.concat([image_test_df, sa_test], ignore_index=True)

    train_df.to_parquet(
        paths.TRAIN_PARQUET_FILE,
        index=False
    )

    val_df.to_parquet(
        paths.VAL_PARQUET_FILE,
        index=False
    )

    test_df.to_parquet(
        paths.TEST_PARQUET_FILE,
        index=False
    )

    train_df.to_csv(
        paths.TRAIN_CSV_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    val_df.to_csv(
        paths.VAL_CSV_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    test_df.to_csv(
        paths.TEST_CSV_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    image_train_df.to_parquet(
        paths.IMAGE_TRAIN_PARQUET_FILE,
        index=False
    )

    image_val_df.to_parquet(
        paths.IMAGE_VAL_PARQUET_FILE,
        index=False
    )

    image_test_df.to_parquet(
        paths.IMAGE_TEST_PARQUET_FILE,
        index=False
    )

    print(
        "\n========== SPLIT =========="
    )

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    print(
        f"Test: {len(test_df)}"
    )

    print(
        f"Total: {len(df)}"
    )


if __name__ == "__main__":

    split_dataset()