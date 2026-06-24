from pathlib import Path

from data.ocr.easyocr_engine import (
    EasyOCREngine
)

from data.loaders.image_loader import (
    discover_images
)


class OCRHandle:

    def __init__(self):

        self.ocr = EasyOCREngine()

    def process_post(
        self,
        post_dir: Path
    ):

        image_files = discover_images(
            post_dir
        )

        if not image_files:

            return False

        texts = []

        for image_path in image_files:

            try:

                text = (
                    self.ocr.extract_text(
                        str(image_path)
                    )
                )

                if text.strip():

                    texts.append(
                        text
                    )

            except Exception as ex:

                print(
                    f"[OCR ERROR] "
                    f"{image_path}: {ex}"
                )

        if not texts:

            return False

        ocr_file = (
            post_dir /
            "ocr.txt"
        )

        with open(
            ocr_file,
            "w",
            encoding="utf8"
        ) as f:

            f.write(
                "\n".join(texts)
            )

        return True

    def process_all(
        self,
        raw_posts_dir
    ):

        raw_posts_dir = Path(
            raw_posts_dir
        )

        total = 0

        for post_dir in raw_posts_dir.iterdir():

            if not post_dir.is_dir():

                continue

            if self.process_post(
                post_dir
            ):

                total += 1

        print(
            f"OCR completed "
            f"for {total} posts."
        )