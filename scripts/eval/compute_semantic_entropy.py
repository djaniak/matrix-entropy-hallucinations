import json
from pathlib import Path
from typing import Dict, List

import torch
import typer
from tqdm import tqdm

from hallucinations.metrics.semantic_entropy import (
    EntailmentDeberta,
    cluster_assignment_entropy,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)
from hallucinations.utils.misc import save_json


def main(
    results_dir: Path = typer.Option(
        ...,
        help="Path to the directory containing the results JSON file",
    ),
) -> None:
    """
    Compute semantic and predictive entropy for model responses.

    The answers file should contain a list of dictionaries with:
    - high_temperature: List of model responses at high temperature
    - low_temperature: Most likely response at low temperature
    - log_likelihoods: Log probabilities for high temperature responses
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entailment_model = EntailmentDeberta(device)

    # Load answers and get shards
    answers_path = results_dir / "answers.json"
    with answers_path.open("r") as f:
        answers = json.load(f)

    # Load high temperature shards (needed for log likelihoods)
    high_temp_shards = list((results_dir / "activations").glob("*high_temperature.pt"))

    results = []
    for i, shard_path in tqdm(
        enumerate(high_temp_shards), desc="Computing semantic entropy", total=len(high_temp_shards)
    ):
        shard = torch.load(shard_path, weights_only=True, mmap=True, map_location="cpu")
        logliks_shard = shard["log_likelihoods"]
        answers_shard = answers[i * len(logliks_shard) : (i + 1) * len(logliks_shard)]

        shard_results = process_shard(logliks_shard, answers_shard, entailment_model)
        results.extend(shard_results)

    output_file = results_dir / "entropy_metrics.json"
    save_json(output_file, results)


def process_shard(
    logliks_shard: torch.Tensor, answers_shard: List[Dict], entailment_model: EntailmentDeberta
) -> List[Dict]:
    """Process a single shard of data."""
    results = []

    for answer, log_likelihoods in zip(answers_shard, logliks_shard):
        entropies = compute_entropies(
            responses=answer["high_temperature"],
            log_likelihoods=log_likelihoods,
            question=answer["question"],
            most_likely_answer=answer["low_temperature"],
            model=entailment_model,
        )

        results.append(
            {
                "question": answer["question"],
                "high_temp_responses": answer["high_temperature"],
                "low_temp_response": answer["low_temperature"],
                **entropies,
            }
        )

    return results


def compute_entropies(
    responses: List[str],
    log_likelihoods: torch.Tensor,
    question: str,
    most_likely_answer: str,
    model: EntailmentDeberta,
) -> Dict[str, float]:
    """Compute semantic and predictive entropy for a set of responses."""
    semantic_ids = get_semantic_ids(
        strings_list=responses,
        model=model,
        strict_entailment=False,
        example={
            "question": question,
            "responses": responses,
            "most_likely_answer": most_likely_answer,
        },
    )

    cluster_entropy = cluster_assignment_entropy(semantic_ids)

    # Aggregate log likelihoods with and without filtering zeros
    log_liks_filtered = aggregate_log_likelihoods(log_likelihoods, filter_zeros=True)
    log_liks_unfiltered = aggregate_log_likelihoods(log_likelihoods, filter_zeros=False)

    # Calculate predictive entropies
    pe_filtered = predictive_entropy(log_liks_filtered)
    pe_unfiltered = predictive_entropy(log_liks_unfiltered)

    # Calculate semantic entropies
    log_likelihood_per_semantic_id_filtered = logsumexp_by_id(
        semantic_ids, log_liks_filtered, agg="sum_normalized"
    )
    log_likelihood_per_semantic_id_unfiltered = logsumexp_by_id(
        semantic_ids, log_liks_unfiltered, agg="sum_normalized"
    )
    se_filtered = predictive_entropy_rao(log_likelihood_per_semantic_id_filtered)
    se_unfiltered = predictive_entropy_rao(log_likelihood_per_semantic_id_unfiltered)

    return {
        "cluster_assignment_entropy": float(cluster_entropy),
        "predictive_entropy_filtered": float(pe_filtered),
        "predictive_entropy_unfiltered": float(pe_unfiltered),
        "semantic_entropy_filtered": float(se_filtered),
        "semantic_entropy_unfiltered": float(se_unfiltered),
    }


def aggregate_log_likelihoods(
    log_likelihoods: torch.Tensor, filter_zeros: bool = True
) -> List[float]:
    """Aggregate log likelihoods by taking mean, optionally filtering zeros and -inf."""
    if filter_zeros:
        filtered = [
            [x for x in log_lik if x not in (float("-inf"), 0.0)] for log_lik in log_likelihoods
        ]
    else:
        filtered = [[x for x in log_lik if x != float("-inf")] for log_lik in log_likelihoods]
    return [torch.mean(torch.tensor(log_lik)).item() for log_lik in filtered]


if __name__ == "__main__":
    typer.run(main)
