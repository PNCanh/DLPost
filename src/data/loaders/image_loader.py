from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = (

    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp",
    "*.bmp"
)


def discover_images(
    post_dir
):

    post_dir = Path(
        post_dir
    )

    images = []

    for ext in (
        SUPPORTED_IMAGE_EXTENSIONS
    ):

        images.extend(
            post_dir.glob(ext)
        )

    return sorted(
        [
            str(img)
            for img in images
        ]
    )


def get_primary_image(
    post_dir
):

    images = discover_images(
        post_dir
    )

    if len(images) == 0:

        return ""

    return images[0]


def get_image_count(
    post_dir
):

    return len(
        discover_images(
            post_dir
        )
    )