from datasets import Dataset, DatasetDict, load_dataset

from hallucinations.config import (
    CcPromptConfig,
    CsvDatasetConfig,
    DatasetConfig,
    MMLUDatasetConfig,
    MMLUPromptConfig,
    PromptConfig,
    QaDatasetConfig,
    QaPromptConfig,
    SquadDatasetConfig,
    TriviaQaDatasetConfig,
)
from hallucinations.datasets.cc import load_custom_claim_dataset, prepare_common_claim_labels
from hallucinations.datasets.formatter import CommonClaimFormatter, MMLUFormatter, QaFormatter


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
            function=QaFormatter(prompt=prompt_config, use_output=use_output),
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
    elif dataset_config.name == "cais/mmlu":
        assert isinstance(prompt_config, MMLUPromptConfig)
        assert isinstance(dataset_config, MMLUDatasetConfig)
        if dataset_config.shuffle:
            dataset = dataset.shuffle(seed=seed)
        if dataset_config.subsample_ratio is not None:
            dataset = dataset.select(range(int(len(dataset) * dataset_config.subsample_ratio)))
        formatted_ds = dataset.map(
            function=MMLUFormatter(prompt=prompt_config),
            batched=False,
            desc="Formatting dataset",
        )

    elif dataset_config.name == "TimoImhof/TriviaQA-in-SQuAD-format":
        assert isinstance(prompt_config, QaPromptConfig)
        assert isinstance(dataset_config, TriviaQaDatasetConfig)
        # Fix answers format
        dataset = dataset.map(lambda x: {"answers": x["answers"]["text"]})
        # Split dataset to obtain similar number of examples as in NQ-Open
        dataset = dataset.train_test_split(test_size=dataset_config.test_split_ratio)["test"]
        formatted_ds = dataset.map(
            function=QaFormatter(prompt=prompt_config, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )

    elif dataset_config.name == "squad_v2":
        assert isinstance(prompt_config, QaPromptConfig)
        assert isinstance(dataset_config, SquadDatasetConfig)
        # Fix answers format
        dataset = dataset.map(lambda x: {"answers": x["answers"]["text"]})
        # Filter out examples without answers
        dataset = dataset.filter(lambda x: len(x["answers"]) > 0)
        # Split dataset to obtain similar number of examples as in NQ-Open
        dataset = dataset.train_test_split(test_size=dataset_config.test_split_ratio)["test"]
        formatted_ds = dataset.map(
            function=QaFormatter(prompt=prompt_config, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )

    if return_raw:
        return dataset, formatted_ds
    else:
        return formatted_ds


def get_dataset(config: DatasetConfig, split: str | None) -> Dataset | DatasetDict:
    if config.name == "google-research-datasets/nq_open":
        assert isinstance(config, QaDatasetConfig)
        return load_dataset(config.name, split=split)
    elif config.name == "TimoImhof/TriviaQA-in-SQuAD-format":
        assert isinstance(config, TriviaQaDatasetConfig)
        return load_dataset(config.name, split=split)
    elif config.name == "squad_v2":
        assert isinstance(config, SquadDatasetConfig)
        return load_dataset(config.name, split=split)
    elif config.name == "custom/common_claim":
        assert isinstance(config, CsvDatasetConfig)
        return load_custom_claim_dataset(config.local_dataset_path, config.dataset_url)
    elif config.name == "cais/mmlu":
        assert isinstance(config, MMLUDatasetConfig)
        print(config.name, config.subset, split)
        return load_dataset(config.name, name=config.subset, split=split)
    else:
        raise ValueError(f"Unknown dataset: {config.name}")
