import copy
import os
from pprint import pformat

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from torch_geometric import seed_everything

from hallucinations.config import GenerateActivationsConfig
from hallucinations.datasets.factory import prepare_dataset
from hallucinations.llm.activation_storage import AllActivationsStorage
from hallucinations.llm.factory import get_llm
from hallucinations.llm.predict import predict_with_llm
from hallucinations.llm.preprocessing import SimpleEncoder
from hallucinations.utils import resolve_config, save_json, save_yaml

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
        split=config.split,
        prompt_config=config.prompt,
        use_output=False,
        return_raw=True,
        seed=config.random_seed,
    )
    # NOTE: Changin order of the dataset by length might cause ordering issues when saving activations

    model_pack = get_llm(
        config.llm,
        device_map="auto",  # loads model in a balanced mode on all available GPUs
    )

    assert not any(key in model_pack.generate_kwargs for key in config.generation_config)
    generation_config = copy.deepcopy(model_pack.llm.generation_config)
    generation_config.update(**(config.generation_config | model_pack.generate_kwargs))

    if config.llm.compile:
        # NOTE: using built-in hf compile results in wall of warnings
        model_pack.llm = torch.compile(model_pack.llm)

    dataset.set_transform(SimpleEncoder(model_pack.tokenizer))

    activation_storage = AllActivationsStorage(config.activations_dir)
    preds = predict_with_llm(
        model=model_pack.llm,
        tokenizer=model_pack.tokenizer,
        dataset=dataset,
        generation_config=generation_config,
        activation_storage=activation_storage,
        batch_size=config.batch_size,
        num_proc=NUM_PROC,
    )

    golds = [ans for ans in raw_ds[config.dataset.target_column_name]]
    results = [{"prediction": ans, "gold": g} for ans, g in zip(preds, golds, strict=True)]

    activation_storage.flush()
    save_json(config.answers_file, results)
    save_yaml(config.config_file, config.model_dump())


if __name__ == "__main__":
    main()
