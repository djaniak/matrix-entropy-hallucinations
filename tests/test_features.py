from typing import Any

import pytest
import torch

from hallucinations.features.hidden_states import (
    Features,
    concat_shard_features,
    select_features_from_layers,
    select_features_from_single_layer,
)


@pytest.fixture
def shard() -> dict[str, Any]:
    # assume batch_size = 1
    # hidden_states.shape = num_layers x [batch_size, seq_len, hidden_dim]
    return {
        "hidden_states": [
            torch.tensor(
                [
                    [
                        [10, 10, 10],
                        [11, 11, 11],
                        [12, 12, 12],
                        [13, 13, 13],
                        [14, 14, 14],
                        [15, 15, 15],
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [20, 20, 20],
                        [21, 21, 21],
                        [22, 22, 22],
                        [23, 23, 23],
                        [24, 24, 24],
                        [25, 25, 25],
                    ]
                ]
            ),
        ],
        "input_length": 4,  # in fact, it's 3 since we don't have hidden states for the first token
        "special_token_mask": torch.tensor([[1, 0, 1, 0, 0, 0, 1]]),
        "decoder_token_mask": torch.tensor([[1, 0, 1, 0, 1, 0, 1]]),
    }


def test_select_hs_last_input_token(shard: dict[str, Any]) -> None:
    features = Features(layer=0, hs_last_input_token=True, hs_last_generated_token=False)
    results = select_features_from_single_layer(shard, layer_idx=0, features=features)

    assert list(results.keys()) == ["hs_last_input_token"]
    assert torch.allclose(results["hs_last_input_token"], torch.tensor([[12, 12, 12]]))


def test_select_hs_last_generated_token(shard: dict[str, Any]) -> None:
    features = Features(layer=0, hs_last_input_token=False, hs_last_generated_token=True)
    results = select_features_from_single_layer(shard, layer_idx=0, features=features)

    assert list(results.keys()) == ["hs_last_generated_token"]
    assert torch.allclose(results["hs_last_generated_token"], torch.tensor([[14, 14, 14]]))


def test_select_features_from_layers(shard: dict[str, Any]) -> None:
    features = Features(layer="all", hs_last_input_token=True, hs_last_generated_token=True)
    results = select_features_from_layers(shard=shard, features=features)

    assert list(results.keys()) == ["hs_last_input_token", "hs_last_generated_token"]
    assert len(results["hs_last_input_token"]) == 2

    target_hs_last_input_token = [torch.tensor([[12, 12, 12]]), torch.tensor([[22, 22, 22]])]
    for target, res in zip(target_hs_last_input_token, results["hs_last_input_token"]):
        assert torch.allclose(target, res)

    target_hs_last_generated_token = [torch.tensor([[14, 14, 14]]), torch.tensor([[24, 24, 24]])]
    for target, res in zip(target_hs_last_generated_token, results["hs_last_generated_token"]):
        assert torch.allclose(target, res)


def test_select_all_features(shard: dict[str, Any]) -> None:
    features = Features(layer=1, hs_last_input_token=True, hs_last_generated_token=True)
    results = select_features_from_single_layer(shard, layer_idx=1, features=features)

    assert list(results.keys()) == ["hs_last_input_token", "hs_last_generated_token"]
    assert torch.allclose(results["hs_last_input_token"], torch.tensor([[22, 22, 22]]))
    assert torch.allclose(results["hs_last_generated_token"], torch.tensor([[24, 24, 24]]))


def test_concat_shard_features(shard: dict[str, Any]) -> None:
    features = Features(layer="all", hs_last_input_token=True, hs_last_generated_token=True)
    # we repeat single shard to simulate multiple shards
    shard_acts = [select_features_from_layers(shard=shard, features=features)] * 2
    results = concat_shard_features(shard_acts)

    assert list(results.keys()) == ["hs_last_input_token", "hs_last_generated_token"]
    # 2 layers
    assert all(len(feats) == 2 for feats in results.values())
    # 2 shards with batch_size = 1 and hidden_dim=3, hence shape is [2, 3]
    for feat_idx in range(2):
        assert all(feats[feat_idx].shape == torch.Size([2, 3]) for feats in results.values())

    # check embedding values are stacked properly [batch_size, hidden_dim]
    assert torch.allclose(
        results["hs_last_input_token"][0], torch.tensor([[12, 12, 12], [12, 12, 12]])
    )
    assert torch.allclose(
        results["hs_last_input_token"][1], torch.tensor([[22, 22, 22], [22, 22, 22]])
    )
    assert torch.allclose(
        results["hs_last_generated_token"][0], torch.tensor([[14, 14, 14], [14, 14, 14]])
    )
    assert torch.allclose(
        results["hs_last_generated_token"][1], torch.tensor([[24, 24, 24], [24, 24, 24]])
    )
