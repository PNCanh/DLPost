from pathlib import Path

import pandas as pd

from configs import paths

from data.builders.explanation_encoder import (
    load_explanation_mapping
)

from data.builders.dataset_validator import (
    DatasetValidator
)



from data.loaders.post_loader import (
    load_post_json
)

from data.loaders.text_loader import (
    load_text,
    load_ocr_text
)

from data.loaders.image_loader import (
    discover_images
)

from data.preprocessing.text_preprocessor import (
    TextPreprocessor
)

from data.builders.label_encoder import (
    load_multiclass_labels
)

from data.builders.feature_extractor import (
    load_keywords,
    keyword_hit_count,
    has_url,
    has_email,
    has_phone
)

def build_record(
    post_dir,
    processor,
    labels,
    explanation_mapping,
    keywords
):

    post = load_post_json(
        post_dir
    )

    DatasetValidator.validate_label(
    post,
    labels
    )

    if post is None:

        return None

    text = load_text(
        post_dir
    )

    ocr_text = load_ocr_text(
        post_dir
    )

    clean_text = (
        processor.process(text)
        if text
        else ""
    )

    clean_ocr = (
        processor.process(ocr_text)
        if ocr_text
        else ""
    )

    full_text = (
        clean_text
        + " [OCR] "
        + clean_ocr
    ).strip()

    text_length = len(
        clean_text.split()
    )

    ocr_length = len(
        clean_ocr.split()
    )

    full_length = len(
        full_text.split()
    )

    images = discover_images(
        post_dir
    )

    image_count = len(
        images
    )

    primary_image = (
        images[0]
        if image_count > 0
        else ""
    )

    ocr_ratio = (
        len(clean_ocr)
        / max(
            len(clean_text),
            1
        )
    )

    has_text = int(
        text_length > 0
    )

    has_ocr_text = int(
        ocr_length > 0
    )

    has_image = int(
        image_count > 0
    )

    label_name = post.get(
        "multi_label",
        "suspicious"
    )

    label_info = labels.get(
        label_name,
        labels["suspicious"]
    )

    multi_label = (
        label_info["id"]
    )

    label_description = (
        label_info["description"]
    )

    risk_level = (
        label_info["risk_level"]
    )

    explanation_labels = (
        post.get(
            "explanation_labels",
            []
        )
    )

    unknown_labels = [
    label
    for label in explanation_labels
    if label not in explanation_mapping
]

    if unknown_labels:

      raise ValueError(
        f"Unknown explanation labels: "
        f"{unknown_labels}"
    )

    explanation_vector = (
        [0]
        * len(
            explanation_mapping
        )
    )

    for label in explanation_labels:

        if label in explanation_mapping:

            explanation_vector[
                explanation_mapping[
                    label
                ]
            ] = 1

    return {

        "post_id":
            post.get(
                "post_id",
                ""
            ),

        "text_clean":
            clean_text,

        "ocr_text":
            clean_ocr,

        "full_text":
            full_text,

        "primary_image":
            primary_image,

        "image_paths":
            images,

        "image_count":
            image_count,

        "text_length":
            text_length,

        "ocr_length":
            ocr_length,

        "full_length":
            full_length,

        "ocr_ratio":
            ocr_ratio,

        "has_text":
            has_text,

        "has_ocr_text":
            has_ocr_text,

        "has_image":
            has_image,

        "keyword_hit_count":
            keyword_hit_count(
                full_text,
                keywords
            ),

        "keyword_vector":
            __import__('data.builders.feature_extractor', fromlist=['']).extract_keyword_vector(
                full_text,
                keywords
            ),

        "has_url":
            has_url(
                full_text
            ),

        "has_email":
            has_email(
                full_text
            ),

        "has_phone":
            has_phone(
                full_text
            ),

        "binary_label":
            int(
                post.get(
                    "binary_label",
                    0
                )
            ),

        "multi_label":
            multi_label,

        "label_name":
            label_name,

        "label_description":
            label_description,

        "risk_level":
            risk_level,

        "explanation_labels":
            explanation_labels,

        "explanation_vector":
            explanation_vector
    }

def build_dataset():

    print(
        "\nLoading resources..."
    )

    processor = TextPreprocessor(
        paths.TEENCODE_FILE,
        paths.ABBREVIATIONS_FILE
    )

    labels = (
        load_multiclass_labels()
    )
    explanation_mapping = (
        load_explanation_mapping()
    )
    keywords = (
        load_keywords()
    )

    raw_dir = paths.RAW_DIR

    output_dir = paths.PROCESSED_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    post_dirs = sorted(
        [
            p
            for p in raw_dir.iterdir()
            if p.is_dir()
        ]
    )

    print(
        f"Found {len(post_dirs)} posts"
    )

    for post_dir in post_dirs:

        try:

            record = build_record(

            post_dir,

            processor,

            labels,

            explanation_mapping,

            keywords
            )

            if record is not None:

                records.append(
                    record
                )

        except Exception as ex:

            print(
                f"[BUILD ERROR] "
                f"{post_dir.name}: "
                f"{ex}"
            )

    if len(records) == 0:

        raise ValueError(
            "No valid records found."
        )

    df = pd.DataFrame(
        records
    )

    output_path = paths.PROCESSED_DATASET_FILE

    df.to_parquet(
        output_path,
        index=False
    )

    print(
        f"\nSaved {len(df)} records"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        "\nBuild completed."
    )
if __name__ == "__main__":

    build_dataset()