class DatasetValidator:

    @staticmethod
    def validate_label(
        post,
        labels
    ):

        label = post.get(
            "multi_label"
        )

        if label not in labels:

            raise ValueError(
                f"Unknown label: {label}"
            )

    @staticmethod
    def validate_binary_label(
        post
    ):

        binary_label = post.get(
            "binary_label"
        )

        if binary_label not in [0, 1]:

            raise ValueError(
                f"Invalid binary label: "
                f"{binary_label}"
            )