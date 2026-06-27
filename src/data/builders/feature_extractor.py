import json
import re

from configs import paths


def load_keywords():

    with open(
        paths.KEYWORDS_FILE,
        encoding="utf8"
    ) as f:

        return json.load(f)


def extract_keyword_vector(
    text,
    keywords
):

    text = text.lower()
    vector = []

    for category, rules in keywords.items():
        if "with_value" in rules:
            for key, values in rules["with_value"].items():
                hit = 0
                for value in values:
                    # Match key.*value pattern (with optional spaces/characters in between, but keep it close)
                    pattern = rf"{re.escape(key.lower())}.{{0,20}}?{re.escape(value.lower())}"
                    if re.search(pattern, text):
                        hit = 1
                        break
                vector.append(hit)
                
        if "without_value" in rules:
            for keyword in rules["without_value"]:
                hit = 0
                pattern = rf"\b{re.escape(keyword.lower().strip())}\b"
                if re.search(pattern, text):
                    hit = 1
                vector.append(hit)

    return vector

def keyword_hit_count(text, keywords):
    return sum(extract_keyword_vector(text, keywords))


def has_url(
    text
):

    pattern = (

        r"(https?:\/\/\S+)"

        r"|"

        r"(www\.\S+)"
    )

    return int(

        bool(

            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
        )
    )


def has_email(
    text
):

    pattern = (
        r"\S+@\S+\.\S+"
    )

    return int(

        bool(

            re.search(
                pattern,
                text
            )
        )
    )


def has_phone(
    text
):

    pattern = (

        r"(0\d{9,10})"

        r"|"

        r"(\+84\d{8,10})"
    )

    return int(

        bool(

            re.search(
                pattern,
                text
            )
        )
    )


def extract_binary_features(
    text,
    keywords
):

    return {

        "keyword_hit_count":
            keyword_hit_count(
                text,
                keywords
            ),

        "has_url":
            has_url(
                text
            ),

        "has_email":
            has_email(
                text
            ),

        "has_phone":
            has_phone(
                text
            )
    }