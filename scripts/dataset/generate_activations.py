import json
import os
from pprint import pformat

import hydra
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig
from torch_geometric import seed_everything
from transformers import GenerationConfig

from hallucinations.config import GenerateActivationsConfig
from hallucinations.datasets.factory import prepare_dataset
from hallucinations.llm.factory import get_llm
from hallucinations.llm.predict import predict_with_llm
from hallucinations.llm.preprocessing import SimpleEncoder
from hallucinations.metrics import compute_squad_metrics
from hallucinations.utils.misc import resolve_config

NUM_PROC = int(os.getenv("NUM_PROC", 1))


@hydra.main(version_base="1.3", config_path="../../config", config_name="generate_activations")
def main(cfg: DictConfig) -> None:
    config = GenerateActivationsConfig(**resolve_config(cfg))
    logger.info(f"Config: {pformat(config.model_dump())}")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.activations_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.random_seed)

    raw_ds, dataset = prepare_dataset(
        dataset_config=config.dataset,
        split=config.dataset.split_name,
        prompt_config=config.prompt,
        use_output=False,
        return_raw=True,
    )
    # NOTE: Sorting dataset by length might cause ordering issues when saving activations

    model_pack = get_llm(
        config.llm,
        device_map="auto",  # loads model in a balanced mode on all available GPUs
    )

    assert not any(key in model_pack.generate_kwargs for key in config.generation_config)
    generation_config = GenerationConfig(**(model_pack.generate_kwargs | config.generation_config))

    model_pack.llm.eval()
    if config.llm.compile:
        # NOTE: using built-in hf compile results in wall of warnings
        model_pack.llm = torch.compile(model_pack.llm)

    dataset.set_transform(SimpleEncoder(model_pack.tokenizer))

    outs = predict_with_llm(
        model=model_pack.llm,
        tokenizer=model_pack.tokenizer,
        dataset=dataset,
        generation_config=generation_config,
        batch_size=config.batch_size,
        num_proc=NUM_PROC,
        activations_save_dir=config.activations_dir,
    )

    golds = [ans for ans in raw_ds["answer"]]
    results = [{"answer": ans, "gold": g} for ans, g in zip(outs, golds, strict=True)]
    metrics = compute_squad_metrics(outs, golds, return_reduced=True, return_all=True)

    with (config.metrics_file).open("w") as file:
        json.dump(metrics, file, indent="\t")

    with (config.answers_file).open("w") as file:
        json.dump(results, file, indent="\t")

    with (config.config_file).open("w") as file:
        yaml.dump(config.model_dump(), file)


if __name__ == "__main__":
    main()
