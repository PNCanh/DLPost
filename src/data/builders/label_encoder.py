import json

from configs import paths


def load_binary_labels():
    with open(
        paths.LABELS_FILE,
        encoding="utf8"
    ) as f:

        data = json.load(f)

    return data["binary"]


def load_multiclass_labels():
    with open(
        paths.LABELS_FILE,
        encoding="utf8"
    ) as f:

        data = json.load(f)

    return data["multiclass"]

def load_label_mapping():
    with open(
        paths.LABELS_FILE,
        encoding="utf8"
    ) as f:

        data = json.load(f)

    mapping = {}

    for label_name, label_info in (
        data["multiclass"].items()
    ):

        mapping[
            label_name
        ] = label_info["id"]

    return mapping

def load_label_metadata():
    with open(
        paths.LABELS_FILE,
        encoding="utf8"
    ) as f:

        data = json.load(f)

    metadata = {}

    for label_name, label_info in (
        data["multiclass"].items()
    ):

        metadata[
            label_info["id"]
        ] = {
            "name": label_name,
            "description": label_info[
                "description"
            ],
            "risk_level": label_info[
                "risk_level"
            ]
        }

    return metadata