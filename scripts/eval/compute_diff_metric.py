from pathlib import Path

import torch
import typer

from hallucinations.metrics import (
    compute_erank,
    compute_logdet,
    compute_metrics_from_shards,
    compute_nesum,
)

METRIC_FN_MAP = {
    "logdet": compute_logdet,
    "erank": compute_erank,
    "nesum": compute_nesum,
}


def main(
    results_dir: Path = typer.Option(
        ..., help="Path to the directory containing the trained results JSON file"
    ),
    results_dir_untrained: Path = typer.Option(
        ..., help="Path to the directory containing the untrained results JSON file"
    ),
    metric: str = typer.Option(
        ..., help="Metric to compute (logdet, nesum, erank, diff_erank, matrix_entropy)"
    ),
    diff_metric_output_path: Path = typer.Option(
        ..., help="Path to the directory to save the diff metric output file"
    ),
    metric_output_path: Path = typer.Option(
        None, help="Path to the directory to save the metric output file"
    ),
    use_token_mask: bool = typer.Option(
        True, "--use_token_mask/--no_token_mask", help="Whether to use special tokens mask"
    ),
) -> None:
    """
    Compute metrics for hidden states across model layers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        metric_fn = METRIC_FN_MAP[metric]
    except KeyError:
        raise ValueError(f"Unknown metric: {metric}. Must be one of {list(METRIC_FN_MAP.keys())}")

    shard_paths_trained = list((results_dir / "activations").glob("*.pt"))
    metric_values_trained = compute_metrics_from_shards(
        shard_paths=shard_paths_trained,
        metric_fn=metric_fn,
        use_token_mask=use_token_mask,
        device=device,
    )
    if metric_output_path is not None:
        torch.save(metric_values_trained, metric_output_path)
        print(f"Metric '{metric}' has been computed and saved to {metric_output_path}")

    shard_paths_untrained = list((results_dir_untrained / "activations").glob("*.pt"))
    metric_values_untrained = compute_metrics_from_shards(
        shard_paths=shard_paths_untrained,
        metric_fn=metric_fn,
        use_token_mask=use_token_mask,
        device=device,
    )
    diff_metric = metric_values_untrained - metric_values_trained
    torch.save(diff_metric, diff_metric_output_path)
    print(f"Diff '{metric}' metric tensor has been saved to {diff_metric_output_path}")


if __name__ == "__main__":
    typer.run(main)
