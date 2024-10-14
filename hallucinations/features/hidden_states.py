from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from tqdm import tqdm


@dataclass
class Features:
    layer: Literal["all", "last"] | int
    hs_last_input_token: bool
    hs_last_generated_token: bool


def load_hidden_state_features_from_shards(
    activations_dir: Path, features: Features
) -> dict[str, list[torch.Tensor]]:
    shard_paths = list(activations_dir.glob("*.pt"))
    if not shard_paths:
        raise ValueError(f"No hidden states found in {activations_dir}")

    shard_acts = []
    for s_path in tqdm(shard_paths, desc="Loading hidden states"):
        shard: dict[str, Any] = torch.load(
            s_path,
            weights_only=True,
            mmap=True,
            map_location="cpu",
        )
        shard_layerwise_acts = select_features_from_layers(shard=shard, features=features)
        shard_acts.append(shard_layerwise_acts)

    return concat_shard_features(shard_acts)


def select_features_from_layers(
    shard: dict[str, Any],
    features: Features,
) -> dict[str, list[torch.Tensor]]:
    if features.layer == "all":
        layer_idx = list(range(len(shard["hidden_states"])))
    elif features.layer == "last":
        layer_idx = [-1]
    else:
        layer_idx = [features.layer]

    shard_layerwise_acts = defaultdict(list)
    for l_idx in layer_idx:
        layer_feats = select_features_from_single_layer(
            shard=shard,
            layer_idx=l_idx,
            features=features,
        )
        for feat_name, layerwise_data in layer_feats.items():
            shard_layerwise_acts[feat_name].append(layerwise_data)

    return dict(shard_layerwise_acts)


def select_features_from_single_layer(
    shard: dict[str, Any],
    layer_idx: int,
    features: Features,
) -> dict[str, torch.Tensor]:
    token_mask = ~torch.bitwise_or(shard["special_token_mask"], shard["decoder_token_mask"])
    # We don't have hidden state for the first token (which is bos token), so we remove it from mask
    token_mask = token_mask[:, 1:]
    # We make input length shorter (see above comment)
    input_length = shard["input_length"] - 1
    layer_hidden_states = shard["hidden_states"][layer_idx]

    feats: dict[str, torch.Tensor] = {}
    if features.hs_last_input_token:
        input_token_mask = token_mask[:, :input_length]
        feats["hs_last_input_token"] = _get_last_masked_token(layer_hidden_states, input_token_mask)
    if features.hs_last_generated_token:
        gen_hs_layer = layer_hidden_states[:, input_length:]
        gen_token_mask = token_mask[:, input_length:]
        feats["hs_last_generated_token"] = _get_last_masked_token(gen_hs_layer, gen_token_mask)
    return feats


def _get_last_masked_token(data: Tensor, mask: Tensor) -> Tensor:
    last_token_idx = (mask.shape[1] - 1) - torch.argmax(
        mask.flip(dims=[1]),
        dim=1,
    )
    # by using gather we copy data thus memory is freed from the large original tensor
    return torch.gather(
        input=data,
        dim=1,
        index=last_token_idx.view(-1, 1, 1).expand(-1, 1, data.size(2)),
    ).squeeze(1)


def concat_shard_features(
    shard_acts: list[dict[str, list[torch.Tensor]]],
) -> dict[str, list[torch.Tensor]]:
    results = defaultdict(list)
    for feat_name in shard_acts[0].keys():
        num_layers = len(shard_acts[0][feat_name])
        for l_idx in range(num_layers):
            results[feat_name].append(
                torch.cat(
                    [current_shard_acts[feat_name][l_idx] for current_shard_acts in shard_acts],
                    dim=0,
                )
            )

    return dict(results)
