from datasets import Dataset, DatasetDict, load_dataset

from hallucinations.config import QaDatasetConfig, QaPromptConfig
from hallucinations.datasets.qa import NqOpenFormatter


def prepare_dataset(
    dataset_config: QaDatasetConfig,
    split: str,
    prompt: QaPromptConfig,
    use_output: bool,
) -> Dataset:
    dataset = get_dataset(config=dataset_config, split=split)
    if dataset_config.name == "google-research-datasets/nq_open":
        dataset = dataset.map(
            function=NqOpenFormatter(prompt=prompt, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )
    return dataset


def get_dataset(config: QaDatasetConfig, split: str | None) -> Dataset | DatasetDict:
    if config.name == "google-research-datasets/nq_open":
        return load_dataset(config.name, split=split)
    else:
        raise ValueError(f"Unknown dataset: {config.name}")
