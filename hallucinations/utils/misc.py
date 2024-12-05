import json
from importlib import import_module
from pathlib import Path
from typing import Any

import requests
import torch
import yaml
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def load_and_resolve_config(path: Path) -> dict:
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    return resolve_config(cfg)


def resolve_config(config: DictConfig, resolve: bool = True) -> dict:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    config_primitive = OmegaConf.to_container(config, resolve=resolve)
    assert isinstance(config_primitive, dict)
    return config_primitive


def sort_dataset_by_input_length(ds: Dataset, field: str) -> tuple[Dataset, list[int]]:
    """Sorts a dataset by the length of a field.

    Args:
        ds (Dataset): dataset to sort
        field (str): field to sort by

    Returns:
        tuple[Dataset, list[int]]: sorted dataset and the reverse sort index
    """
    item_lenghts = torch.tensor(
        ds.map(lambda item: {"length": len(item[field])}, remove_columns=ds.column_names)["length"]
    )
    sort_idx = torch.argsort(item_lenghts, stable=True, descending=True)
    reverse_sort_idx = torch.argsort(sort_idx, stable=True).tolist()
    return ds.select(sort_idx), reverse_sort_idx


def download_dataset(url: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)

    logger.info(f"Dataset downloaded and saved to {local_path}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent="\t")


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        yaml.dump(data, f)


def import_cls_from_str(name: str) -> type:
    name = name.split(".")  # type: ignore
    module = import_module(".".join(name[:-1]))
    assert hasattr(module, name[-1]), f"Unknown class: {name[-1]}"
    return getattr(module, name[-1])
