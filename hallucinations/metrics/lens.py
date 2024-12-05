from typing import List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer
from tuned_lens.nn.lenses import LogitLens, TunedLens
from tuned_lens.plotting import PredictionTrajectory

from hallucinations.config import GenerateTrajectoriesConfig
from hallucinations.llm.factory import ModelForGeneration

# https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens/meta-llama
AVAILABLE_TUNED_LENS = [
    "Llama-2-13b-chat-hf",
    "Llama-2-13b-hf",
    "Llama-2-7b-chat-hf",
    "Llama-2-7b-hf",
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3-8B",
]


def is_lens_available(config: GenerateTrajectoriesConfig) -> bool:
    model_name = str(config.llm.name).split("/")[-1]
    if config.lens == "tuned":
        return model_name in AVAILABLE_TUNED_LENS
    return True


def get_lens(
    config: GenerateTrajectoriesConfig, model_pack: ModelForGeneration
) -> TunedLens | LogitLens:
    lens_types = {
        "tuned": lambda: TunedLens.from_model_and_pretrained(model_pack.llm).to(
            model_pack.llm.device
        ),
        "logit": lambda: LogitLens.from_model(model_pack.llm),
    }

    if config.lens not in lens_types:
        raise ValueError(f"Unknown lens type: {config.lens}")

    return lens_types[config.lens]()


def _ids_to_tokens(ids: np.ndarray, tokenizer: PreTrainedTokenizer) -> np.ndarray:
    """Convert a numpy array of token IDs to their corresponding tokens using the tokenizer."""
    tokens = tokenizer.convert_ids_to_tokens(ids.flatten().tolist())
    return np.array(tokens).reshape(ids.shape)


def _get_topk_tokens_and_values(
    k: int, sort_by: np.ndarray, values: np.ndarray, tokenizer: PreTrainedTokenizer
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the top-k tokens and their corresponding values, sorted by a given metric."""
    # Get the top-k indices & corresponding probabilities/values for each element
    topk_inds = np.argpartition(sort_by, -k, axis=-1)[..., -k:]
    topk_sort_by = np.take_along_axis(sort_by, topk_inds, axis=-1)
    topk_values = np.take_along_axis(values, topk_inds, axis=-1)

    # Ensure that the top-k tokens are sorted by the given metric in descending order
    sorted_top_k_inds = np.argsort(-topk_sort_by, axis=-1)
    topk_inds = np.take_along_axis(topk_inds, sorted_top_k_inds, axis=-1)
    topk_values = np.take_along_axis(topk_values, sorted_top_k_inds, axis=-1)

    # Convert token IDs to actual tokens
    topk_tokens = _ids_to_tokens(topk_inds, tokenizer)

    return topk_tokens, topk_values, topk_inds


def _calculate_layer_probs(
    target_tokens: List[int], topk_inds: np.ndarray, topk_probs: np.ndarray
) -> np.ndarray:
    """Calculate the sum of probabilities for given tokens across layers at the last token position."""
    num_layers = topk_inds.shape[0]
    layer_probs_tensor = np.zeros(num_layers)

    # Create a mask for target tokens
    target_tokens_set = set(target_tokens)
    mask = np.isin(topk_inds[:, -1], list(target_tokens_set))

    # Use the mask to sum probabilities directly
    layer_probs_tensor = np.sum(topk_probs[:, -1] * mask, axis=-1)

    return layer_probs_tensor


def _get_token_ids_with_label(
    tokenizer: PreTrainedTokenizer, label: str, max_length: int = 6
) -> List[int]:
    """Retrieve token IDs from the tokenizer vocabulary that contain the specified label."""
    label_tokens = {
        tok: ind
        for tok, ind in tokenizer.get_vocab().items()
        if label.lower() in tok.lower() and len(tok) <= max_length
    }

    return list(label_tokens.values())


def compute_pred_traj_statistics(
    pred_traj: PredictionTrajectory,
    tracked_tokens: list[str] | None,
    matched_tokens_max_length: int,
    k: int = 1000,
) -> dict[str, np.ndarray]:
    """
    Compute statistics for prediction trajectories, including layer probabilities for
    true/false tokens, layer entropies, forward KL divergences, cross-entropies, and
    maximum probabilities.
    """

    def _compute_tracked_tokens_layer_probs(
        pred_traj: PredictionTrajectory,
        tracked_tokens: List[str],
        matched_tokens_max_length: int,
        k: int = 1000,
    ) -> dict[str, np.ndarray]:
        "Get the top-k tokens and corresponding values for tracked tokens"
        topk_tokens, topk_probs, topk_inds = _get_topk_tokens_and_values(
            k=k, sort_by=pred_traj.log_probs, values=pred_traj.probs, tokenizer=pred_traj.tokenizer
        )

        tokens_ids = {}
        for tok in tracked_tokens:
            tokens_ids[tok] = _get_token_ids_with_label(
                pred_traj.tokenizer, label=tok, max_length=matched_tokens_max_length
            )

        # TODO: Add tracing top-1 token if it is outside of given tracked tokens

        tokens_layer_probs = {}
        for tok, token_ids in tokens_ids.items():
            tok_key = f"{tok}_token_prob"
            tokens_layer_probs[tok_key] = _calculate_layer_probs(token_ids, topk_inds, topk_probs)

        return tokens_layer_probs

    def _compute_tokens_stats(pred_traj: PredictionTrajectory) -> dict[str, np.ndarray]:
        "Compute layer entropies, forward KLD, cross-entropies, and maximum probabilities"
        entropy = -np.sum(pred_traj.probs[:, -1] * pred_traj.log_probs[:, -1], axis=-1)

        # model_log_probs = log probs from last layer and last token
        model_log_probs = pred_traj.model_log_probs[..., np.newaxis, :, :][:, -1]
        forward_kl = np.sum(
            np.exp(model_log_probs) * (model_log_probs - pred_traj.log_probs[:, -1]), axis=-1
        )

        def _select_values_along_seq_axis(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
            """Select target values along the vocab dimension."""
            batch_indices = np.arange(values.shape[0])[:, None]
            seq_indices = np.arange(values.shape[1])[None, :]
            return values[batch_indices, seq_indices, targets]

        cross_entropy = -_select_values_along_seq_axis(pred_traj.log_probs, pred_traj.targets)[
            :, -1
        ]
        max_prob = np.exp(pred_traj.log_probs[:, -1].max(-1))

        return {
            "entropy": entropy,
            "forward_kl": forward_kl,
            "cross_entropy": cross_entropy,
            "max_prob": max_prob,
        }

    stats = {}

    if tracked_tokens is not None:
        tracked_tokens_layer_probs = _compute_tracked_tokens_layer_probs(
            pred_traj,
            tracked_tokens=tracked_tokens,
            matched_tokens_max_length=matched_tokens_max_length,
            k=k,
        )
        stats.update(tracked_tokens_layer_probs)

    tokens_stats = _compute_tokens_stats(pred_traj)
    stats.update(tokens_stats)

    return stats


def get_answer_token(pred_traj: PredictionTrajectory) -> str:
    """
    Compute statistics for prediction trajectories, including layer probabilities for
    true/false tokens, layer entropies, forward KL divergences, cross-entropies, and
    maximum probabilities.
    """
    # Get the top-k tokens and corresponding values
    topk_tokens, _, _ = _get_topk_tokens_and_values(
        k=1, sort_by=pred_traj.log_probs, values=pred_traj.probs, tokenizer=pred_traj.tokenizer
    )
    return topk_tokens[-1, -1, 0].item()


def get_pred_traj_stats_names(tracked_tokens: list[str]) -> list[str]:
    """Get the names of the statistics computed for prediction trajectories."""
    stats_names = [
        "entropy",
        "forward_kl",
        "cross_entropy",
        "max_prob",
    ]
    for tok in tracked_tokens:
        stats_names.append(f"{tok}_token_prob")
    return stats_names
