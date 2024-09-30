from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets

from hallucinations.utils.misc import download_dataset

LABEL_MODES = {
    "binary": ["True", "False"],
    "binary_with_neither": ["True or False", "Neither"],
    "multi": ["True", "False", "Neither"],
}


def load_custom_claim_dataset(local_path: str | Path, url: str | None = None) -> Dataset:
    local_path = Path(local_path)
    if not local_path.exists():
        assert url is not None, "Dataset not found and no URL provided"
        download_dataset(url, local_path)

    df = pd.read_csv(local_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    dataset = Dataset.from_pandas(df)

    return dataset


def prepare_common_claim_labels(dataset: Dataset, label_mode: str, seed: int = 42) -> Dataset:
    # Filter out the statements without agreement
    dataset = dataset.filter(lambda x: x["agreement"] == 1.0).remove_columns("agreement")

    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unknown label mode: {label_mode}")

    # Adjust labels if necessary for binary_with_neither
    if label_mode == "binary_with_neither":
        dataset = dataset.map(
            lambda x: {"label": "True or False" if x["label"] in ["True", "False"] else "Neither"}
        )

    # Compute the minimum count for each label to balance the dataset
    min_n = min(
        dataset.filter(lambda x: x["label"] == label).num_rows for label in LABEL_MODES[label_mode]
    )

    # Select equal samples for each label
    balanced_dataset = concatenate_datasets(
        [
            dataset.filter(lambda x: x["label"] == label).select(range(min_n))
            for label in LABEL_MODES[label_mode]
        ]
    ).shuffle(seed=seed)

    return balanced_dataset
