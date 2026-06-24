import json

from data.preprocessing.vietnamese_normalizer import normalize_unicode
from data.preprocessing.emoji_handler import replace_emoji
from data.preprocessing.text_cleaner import clean_text


class TextPreprocessor:

    def __init__(
        self,
        teencode_file,
        abbreviation_file
    ):

        with open(
            teencode_file,
            encoding="utf8"
        ) as f:

            self.teencode = json.load(f)

        with open(
            abbreviation_file,
            encoding="utf8"
        ) as f:

            self.abbreviations = json.load(f)

    def normalize_teencode(
        self,
        text
    ):

        for k, v in self.teencode.items():

            text = text.replace(
                k,
                v
            )

        return text

    def normalize_abbreviation(
        self,
        text
    ):

        for k, v in self.abbreviations.items():

            text = text.replace(
                k,
                v
            )

        return text

    def process(
        self,
        text
    ):

        text = normalize_unicode(
            text
        )

        text = replace_emoji(
            text
        )

        text = self.normalize_teencode(
            text
        )

        text = self.normalize_abbreviation(
            text
        )

        text = clean_text(
            text
        )

        return text.lower()