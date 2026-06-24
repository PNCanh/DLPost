import pandas as pd


def show_dataset_stats(
    parquet_file
):

    df = pd.read_parquet(
        parquet_file
    )

    print("\n==========")
    print("DATASET")
    print("==========")

    print(
        f"Total samples: "
        f"{len(df)}"
    )

    print(
        "\nMulti Label Distribution"
    )

    print(
        df["multi_label"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nBinary Distribution"
    )

    print(
        df["binary_label"]
        .value_counts()
        .sort_index()
    )