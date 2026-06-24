import unicodedata


def normalize_unicode(
    text
):

    return unicodedata.normalize(
        "NFC",
        text
    )