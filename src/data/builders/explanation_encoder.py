import json

from configs import paths


def load_explanation_mapping():

    with open(
        paths.EXPLANATION_LABELS_FILE,
        encoding="utf8"
    ) as f:

        data = json.load(f)

    ids = sorted(

        [
            info["id"]
            for info in data.values()
        ]
    )

    expected_ids = list(

        range(
            len(ids)
        )
    )

    if ids != expected_ids:

        raise ValueError(

            "Explanation label ids "
            "must be continuous "
            "from 0..N-1"
        )

    mapping = {}

    for name, info in data.items():

        mapping[
            name
        ] = info["id"]

    return mapping


def load_explanation_metadata():

    with open(
        paths.EXPLANATION_LABELS_FILE,
        encoding="utf8"
    ) as f:

        data = json.load(f)

    metadata = {}

    for name, info in data.items():

        metadata[
            info["id"]
        ] = {

            "name":
                name,

            "description":
                info.get(
                    "description",
                    ""
                )
        }

    return metadata