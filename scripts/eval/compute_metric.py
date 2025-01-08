from pathlib import Path

import torch
import typer

from hallucinations.metrics import (
    compute_logdet,
    compute_logdet_ver1,
    compute_metrics_from_shards,
    compute_nesum,
)

METRIC_FN_MAP = {
    "logdet": compute_logdet,
    "logdet_ver1": compute_logdet_ver1,
    "nesum": compute_nesum,
}


def main(
    results_dir: Path = typer.Option(
        ...,
        help="Path to the directory containing the results JSON file",
    ),
    output_path: Path = typer.Option(..., help="Path to save the computed metric values"),
    metric: str = typer.Option(
        ..., help="Metric to compute (logdet, nesum, erank, diff_erank, matrix_entropy)"
    ),
    use_token_mask: bool = typer.Option(
        False,
        "--use_token_mask/--no_token_mask",
        help="Whether to use special tokens mask",
    ),
) -> None:
    """
    Compute metrics for hidden states across model layers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shard_paths = list((results_dir / "activations").glob("*.pt"))

    try:
        metric_fn = METRIC_FN_MAP[metric]
    except KeyError:
        raise ValueError(f"Unknown metric: {metric}. Must be one of {list(METRIC_FN_MAP.keys())}")

    metric_values = compute_metrics_from_shards(
        shard_paths=shard_paths,
        metric_fn=metric_fn,
        use_token_mask=use_token_mask,
        device=device,
    )

    torch.save(metric_values, output_path)
    print(f"Metric '{metric}' has been computed and saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
