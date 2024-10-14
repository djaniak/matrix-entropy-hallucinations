from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets

from hallucinations.config import GenerateActivationsConfig, LllmJudgeConfig
from hallucinations.datasets.factory import get_dataset
from hallucinations.utils.misc import load_and_resolve_config, load_json


def load_qa_dataset_with_metrics(answers_dir: Path) -> Dataset:
    config = GenerateActivationsConfig(**load_and_resolve_config(answers_dir / "config.yaml"))
    dataset = get_dataset(config=config.dataset, split=config.split)
    metrics = Dataset.from_pandas(load_qa_metrics(answers_dir, config=config))

    metrics = metrics.remove_columns("answers")

    return concatenate_datasets([dataset, metrics])


def load_qa_metrics(
    answers_dir: Path, config: GenerateActivationsConfig | None = None
) -> pd.DataFrame:
    if config is None:
        config = GenerateActivationsConfig(**load_and_resolve_config(answers_dir / "config.yaml"))

    answers = pd.read_json(config.answers_file)

    metrics = pd.DataFrame(load_json(config.metrics_file)["all"])
    metrics = pd.concat([answers, metrics], axis=1)

    llm_judge_config_file = answers_dir / "llm_judge_config.yaml"
    if llm_judge_config_file.exists():
        llm_judge_raw_config = load_and_resolve_config(llm_judge_config_file)
        llm_judge_config = LllmJudgeConfig(**llm_judge_raw_config)
        llm_judge_results = pd.DataFrame(
            {"llm_as_judge": load_json(llm_judge_config.evaluation_file)}
        )
        metrics = pd.concat([metrics, llm_judge_results], axis=1)

    return metrics
