from datasets import Dataset, DatasetDict, load_dataset

from hallucinations.config import (
    CcPromptConfig,
    CsvDatasetConfig,
    DatasetConfig,
    PromptConfig,
    QaDatasetConfig,
    QaPromptConfig,
)
from hallucinations.datasets.cc import load_custom_claim_dataset, prepare_common_claim_labels
from hallucinations.datasets.formatter import CommonClaimFormatter, NqOpenFormatter


def prepare_dataset(
    dataset_config: DatasetConfig,
    split: str | None,
    prompt_config: PromptConfig,
    use_output: bool,
    seed: int,
    return_raw: bool = False,
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
        return load_custom_claim_dataset(config.local_dataset_path, config.dataset_url)
    else:
        raise ValueError(f"Unknown dataset: {config.name}")
