import json
from pathlib import Path


def load_post_json(post_dir):

    post_file = Path(post_dir) / "post.json"

    if not post_file.exists():

        return None

    try:

        with open(
            post_file,
            encoding="utf8"
        ) as f:

            return json.load(f)

    except Exception as ex:

        print(
            f"[POST ERROR] {post_file}: {ex}"
        )

        return None