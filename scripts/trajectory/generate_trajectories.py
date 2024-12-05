"""
This script generates prediction trajectories using either LogitLens or TunedLens to analyze how model predictions evolve
across transformer layers. LogitLens tracks raw logit values while TunedLens uses learned linear probes to extract
predictions at each layer. These trajectories provide insights into how representations and predictions develop through the model's computation.

References:
- interpreting gpt: the logit lens. LessWrong, 2020. URL https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Belrose et al. (2023) “Eliciting Latent Predictions from Transformers with the Tuned Lens.” https://arxiv.org/abs/2303.08112
"""

import json
from pprint import pformat
from typing import Any

import hydra
import numpy as np
import yaml
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig
from torch_geometric import seed_everything
from tqdm import tqdm
from tuned_lens.nn.lenses import LogitLens, TunedLens
from tuned_lens.plotting import PredictionTrajectory

from hallucinations.config import CcPromptConfig, GenerateTrajectoriesConfig, MMLUPromptConfig
from hallucinations.datasets.factory import prepare_dataset
from hallucinations.llm.factory import ModelForGeneration, get_llm
from hallucinations.llm.preprocessing import SimpleEncoder
from hallucinations.metrics.lens import (
    compute_pred_traj_statistics,
    get_answer_token,
    get_lens,
    is_lens_available,
)
from hallucinations.utils.misc import resolve_config

ProcessingResultItem = tuple[str, dict[str, np.ndarray]]
ProcessingResult = tuple[list[str], dict[str, list]]
FinalResult = list[dict[str, str | list[str]]]


def get_tracked_tokens_and_max_length(config: GenerateTrajectoriesConfig) -> tuple[list[str], int]:
    if config.dataset.name == "custom/common_claim":
        tracked_tokens = ["True", "False"]
        matched_tokens_max_length = 6
    elif config.dataset.name == "cais/mmlu":
        tracked_tokens = ["A", "B", "C", "D"]
        matched_tokens_max_length = 1
    else:
        raise NotImplementedError(f"Tracked tokens not implemented for {config.dataset.name}")

    return tracked_tokens, matched_tokens_max_length


def process_example(
    example: dict[str, Any],
    lens: TunedLens | LogitLens,
    model_pack: ModelForGeneration,
    tracked_tokens: list[str],
    matched_tokens_max_length: int,
    k: int,
) -> ProcessingResultItem:
    input_ids = example["input_ids"].tolist()
    targets = input_ids[1:] + [model_pack.tokenizer.eos_token_id]

    pred_traj = PredictionTrajectory.from_lens_and_model(
        lens=lens,
        model=model_pack.llm,
        input_ids=input_ids,
        tokenizer=model_pack.tokenizer,
        targets=targets,
    )

    answer = get_answer_token(pred_traj)

    stats = compute_pred_traj_statistics(
        pred_traj,
        tracked_tokens=tracked_tokens,
        matched_tokens_max_length=matched_tokens_max_length,
        k=k,
    )

    return answer, stats


def process_dataset(
    dataset: Dataset,
    lens: TunedLens | LogitLens,
    model_pack: ModelForGeneration,
    tracked_tokens: list[str],
    matched_tokens_max_length: int,
    k: int,
) -> ProcessingResult:
    answers, statistics = [], {}

    for index, example in enumerate(tqdm(dataset)):
        answer, stats = process_example(
            example=example,
            lens=lens,
            model_pack=model_pack,
            tracked_tokens=tracked_tokens,
            matched_tokens_max_length=matched_tokens_max_length,
            k=k,
        )

        answers.append(answer)

        for stat_name, stat in stats.items():
            if stat_name not in statistics:
                statistics[stat_name] = [None] * len(dataset)
            statistics[stat_name][index] = stat.tolist()

    return answers, statistics


def create_results(
    raw_ds: Dataset, config: GenerateTrajectoriesConfig, answers: list[str]
) -> FinalResult:
    golds = raw_ds[config.dataset.target_column_name]

    if config.dataset.name == "custom/common_claim":
        assert isinstance(config.prompt, CcPromptConfig)
        inputs = raw_ds[config.prompt.statement_key]
    elif config.dataset.name == "cais/mmlu":
        assert isinstance(config.prompt, MMLUPromptConfig)
        golds = [chr(gold + 97).capitalize() for gold in golds]
        inputs = raw_ds[config.prompt.question_key]
        subjects = raw_ds[config.prompt.subject_key]

    results = []
    for ans, gold, inp in zip(answers, golds, inputs, strict=True):
        result = {
            "answer": ans,
            "gold": gold,
            "inputs": inp,
        }
        results.append(result)

    if config.dataset.name == "cais/mmlu":
        results = [{**res, "subject": subj} for res, subj in zip(results, subjects, strict=True)]

    return results


@hydra.main(version_base="1.3", config_path="../../config", config_name="generate_trajectories")
def main(cfg: DictConfig) -> None:
    config = GenerateTrajectoriesConfig(**resolve_config(cfg))
    logger.info(f"Config: {pformat(config.model_dump())}")

    if not is_lens_available(config):
        logger.error("Lens not available for the model")
        return

    config.results_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.random_seed)

    raw_ds, dataset = prepare_dataset(
        dataset_config=config.dataset,
        split=config.dataset.test_split_name,
        prompt_config=config.prompt,
        use_output=False,
        return_raw=True,
        seed=config.random_seed,
    )
    # NOTE: Sorting dataset by length might cause ordering issues when saving activations

    model_pack = get_llm(
        config.llm,
        device_map="auto",  # loads model in a balanced mode on all available GPUs
    )
    model_pack.llm.eval()

    dataset.set_transform(SimpleEncoder(model_pack.tokenizer))

    lens = get_lens(config, model_pack)

    tracked_tokens, matched_tokens_max_length = get_tracked_tokens_and_max_length(config)

    answers, stats = process_dataset(
        dataset=dataset,
        lens=lens,
        model_pack=model_pack,
        tracked_tokens=tracked_tokens,
        matched_tokens_max_length=matched_tokens_max_length,
        k=config.topk_tokens,
    )

    results = create_results(raw_ds, config, answers)

    with (config.stats_file).open("w") as file:
        json.dump(stats, file, indent="\t")

    with (config.answers_file).open("w") as file:
        json.dump(results, file, indent="\t")

    with (config.config_file).open("w") as file:
        yaml.dump(config.model_dump(), file)


if __name__ == "__main__":
    main()
