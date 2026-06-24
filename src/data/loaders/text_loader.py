from pathlib import Path


def read_file(
    file_path
):

    file_path = Path(
        file_path
    )

    if not file_path.exists():

        return ""

    try:

        with open(
            file_path,
            encoding="utf8"
        ) as f:

            return (
    f.read()
    .replace("\x00", " ")
    .strip()
)

    except Exception as ex:

        print(
            f"[TEXT ERROR] {file_path}: {ex}"
        )

        return ""


def load_text(
    post_dir
):

    return read_file(
        Path(post_dir)
        / "text.txt"
    )


def load_ocr_text(
    post_dir
):

    return read_file(
        Path(post_dir)
        / "ocr.txt"
    )