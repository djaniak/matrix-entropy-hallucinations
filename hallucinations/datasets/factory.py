from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, load_dataset

from hallucinations.config import (
    CcPromptConfig,
    CsvDatasetConfig,
    DatasetConfig,
    PromptConfig,
    QaDatasetConfig,
    QaPromptConfig,
)
from hallucinations.datasets.cc import CommonClaimFormatter, prepare_common_claim_labels
from hallucinations.datasets.qa import NqOpenFormatter


def prepare_dataset(
    dataset_config: DatasetConfig,
    split: str | None,
    prompt_config: PromptConfig,
    use_output: bool,
    return_raw: bool = False,
    seed: int = 42,
) -> Dataset | tuple[Dataset, Dataset]:
    dataset = get_dataset(config=dataset_config, split=split)

    if dataset_config.name == "google-research-datasets/nq_open":
        assert isinstance(prompt_config, QaPromptConfig)
        formatted_ds = dataset.map(
            function=NqOpenFormatter(prompt=prompt_config, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )

    elif dataset_config.name == "custom/common_claim":
        assert isinstance(prompt_config, CcPromptConfig)
        dataset = prepare_common_claim_labels(
            dataset, label_mode=prompt_config.label_mode, seed=seed
        )
        dataset = dataset.rename_column("examples", prompt_config.statement_key)
        dataset = dataset.train_test_split(test_size=0.1)
        formatted_ds = dataset.map(
            function=CommonClaimFormatter(prompt=prompt_config),
            batched=False,
            desc="Formatting dataset",
        )

    if return_raw:
        return dataset, formatted_ds
    else:
        return dataset


def get_dataset(config: DatasetConfig, split: str | None) -> Dataset | DatasetDict:
    if config.name == "google-research-datasets/nq_open":
        assert isinstance(config, QaDatasetConfig)
        return load_dataset(config.name, split=split)
    elif config.name == "custom/common_claim":
        assert isinstance(config, CsvDatasetConfig)
        return load_custom_dataset(config.local_dataset_path, config.dataset_url)
    else:
        raise ValueError(f"Unknown dataset: {config.name}")


def load_custom_dataset(local_path: str | Path, url: str | None = None) -> Dataset:
    local_path = Path(local_path)
    if not local_path.exists():
        assert url is not None, "Dataset not found and no URL provided"
        download_dataset(url, local_path)

    df = pd.read_csv(local_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    dataset = Dataset.from_pandas(df)

    return dataset


def download_dataset(url: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)

    print(f"Dataset downloaded and saved to {local_path}")
