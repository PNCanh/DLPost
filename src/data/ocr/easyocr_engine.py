import torch
import easyocr

class EasyOCREngine:

    def __init__(self):

        use_gpu = torch.cuda.is_available()

        print(
            f"OCR GPU: {use_gpu}"
        )

        self.reader = easyocr.Reader(
            ["vi", "en"],
            gpu=use_gpu
        )

    def extract_text(
        self,
        image_path
    ):

        results = self.reader.readtext(
            image_path,
            detail=0
        )

        return " ".join(results)