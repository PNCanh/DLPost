from pathlib import Path

import pandas as pd

from configs import paths


def load_train_dataset():

    return pd.read_parquet(
        paths.TRAIN_PARQUET_FILE
    )


def load_val_dataset():

    return pd.read_parquet(
        paths.VAL_PARQUET_FILE
    )


def load_test_dataset():

    return pd.read_parquet(
        paths.TEST_PARQUET_FILE
    )


def load_processed_dataset():

    return pd.read_parquet(
        paths.PROCESSED_DATASET_FILE
    )