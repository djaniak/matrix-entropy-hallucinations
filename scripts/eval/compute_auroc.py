import json
import logging
from pathlib import Path

import numpy as np
import torch
import typer
from sklearn.metrics import roc_auc_score

from hallucinations.utils.defaults import (
    ACTIVATIONS_DIR,
    DATASET_NAMES_MAPPING,
    MODEL_NAMES_MAPPING,
    PROMPT_NAMES_MAPPING,
)
from hallucinations.utils.metric_loader import (
    compute_qa_results,
    get_activation_dirs,
    load_qa_answers,
    load_qa_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


METRICS_MAPPING = {
    "diff_erank": "Diff-eRank",
    "erank": "eRank",
    "logdet": "LogDet",
    "semantic_entropy_unfiltered": "Sem-Entropy",
    "predictive_entropy_unfiltered": "Pred-Entropy",
}

SEMANTIC_ENTROPY_METRICS = ["semantic_entropy_unfiltered", "predictive_entropy_unfiltered"]


def main(
    output_dir: Path = typer.Option(..., help="Path to save the computed metric values"),
) -> None:
    logger.info("Computing AUROC for implemented metrics")

    activations_dirs = get_activation_dirs(ACTIVATIONS_DIR)
    model_results = load_qa_results(activations_dirs)

    metrics_results = []
    for metric_name in ["diff_erank", "erank", "logdet"]:
        for result in model_results:
            metrics_path = result["path"].parent / f"{metric_name}.pt"
            if not metrics_path.exists():
                logger.warning(f"Metrics file not found: {metrics_path}")
                continue
            metrics = torch.load(metrics_path, weights_only=True)

            labels = [1 if x["rougeL_fmeasure"] >= 0.3 else 0 for x in result["qa_metrics"]["all"]]

            for idx, layer_metrics in enumerate(metrics):
                if metric_name == "diff_erank":
                    auroc = roc_auc_score(labels, layer_metrics)
                else:
                    auroc = 1 - roc_auc_score(labels, layer_metrics)

                metrics_results.append(
                    {
                        "model": MODEL_NAMES_MAPPING[result["model"]],
                        "prompt": PROMPT_NAMES_MAPPING[result["prompt"]],
                        "temperature": result["temperature"].capitalize(),
                        "dataset": DATASET_NAMES_MAPPING[result["dataset"]],
                        "metric": METRICS_MAPPING[metric_name],
                        "auroc": auroc,
                        "layer": idx + 1,
                    }
                )

    logger.info("Computing AUROC for semantic entropy metrics")
    data_dir = Path("/data/hallucinations/activations")
    activations_dirs = get_activation_dirs(data_dir)
    activations_dirs = [x for x in activations_dirs if "multiple_samples" in str(x["path"])]
    model_answers = load_qa_answers(activations_dirs)
    model_results = compute_qa_results(model_answers, pred_key="low_temperature", gold_key="gold")

    for result in model_results:
        metrics_path = result["path"].parent / "entropy_metrics.json"
        if not metrics_path.exists():
            logger.warning(f"Entropy metrics file not found: {metrics_path}")
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        labels = [1 if x["rougeL_fmeasure"] >= 0.3 else 0 for x in result["qa_metrics"]["all"]]

        for metric_name in ["semantic_entropy_unfiltered", "predictive_entropy_unfiltered"]:
            metric_values = [x[metric_name] for x in metrics]
            auroc = 1 - roc_auc_score(labels, metric_values)
            metrics_results.append(
                {
                    "model": MODEL_NAMES_MAPPING[result["model"]],
                    "prompt": PROMPT_NAMES_MAPPING[result["prompt"]],
                    "temperature": "Low",
                    "dataset": DATASET_NAMES_MAPPING[result["dataset"]],
                    "metric": METRICS_MAPPING[metric_name],
                    "auroc": auroc,
                    "layer": np.inf,
                }
            )

    # sort by dataset, model, prompt, temperature, layer, metric
    logger.info("Sorting results")
    metrics_results.sort(
        key=lambda x: (
            x["dataset"],
            x["model"],
            x["prompt"],
            x["temperature"],
            x["layer"],
            x["metric"],
        )
    )

    print("\nAUROC Results (Best Layer):")
    print("-" * 80)

    # Group results by everything except layer
    grouped_results = {}
    for result in metrics_results:
        key = (
            result["dataset"],
            result["model"],
            result["prompt"],
            result["temperature"],
            result["metric"],
        )
        if key not in grouped_results:
            grouped_results[key] = result
        elif result["auroc"] > grouped_results[key]["auroc"]:
            grouped_results[key] = result

    # Print best results
    for result in sorted(
        grouped_results.values(),
        key=lambda x: (x["dataset"], x["model"], x["prompt"], x["temperature"], x["metric"]),
    ):
        print(
            f"Dataset: {result['dataset']:<15} "
            f"Model: {result['model']:<10} "
            f"Prompt: {result['prompt']:<15} "
            f"Temp: {result['temperature']:<8} "
            f"Layer: {str(result['layer']):<6} "
            f"Metric: {result['metric']:<25} "
            f"AUROC: {result['auroc']:.3f}"
        )
    print("-" * 80)

    logger.info(f"Saving results to {output_dir / 'auroc.json'}")
    with open(output_dir / "auroc.json", "w") as f:
        json.dump(metrics_results, f)

    logger.info("AUROC computation completed")


if __name__ == "__main__":
    typer.run(main)
