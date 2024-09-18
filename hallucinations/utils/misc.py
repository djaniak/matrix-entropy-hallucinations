from pathlib import Path

import torch
from datasets import Dataset
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
