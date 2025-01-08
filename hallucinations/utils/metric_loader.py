import json
from pathlib import Path

import pandas as pd
import torch

from hallucinations.metrics.rouge import compute_rouge_score
from hallucinations.metrics.squad import compute_squad_metrics


def get_activation_dirs(data_dir: Path) -> list[dict]:
    """Get activation directories and metadata.

    Args:
        data_dir: Path to data directory containing activation directories

    Returns:
        List of dictionaries containing activation directory info
    """
    activations_dirs = []

    for activation_dir in data_dir.rglob("activations"):
        record = {
            "model": activation_dir.parent.parent.parent.name,
            "temperature": "low" if "low" in activation_dir.parent.parent.name else "high",
            "prompt": activation_dir.parent.name.split("__")[0],
            "path": activation_dir,
            "dataset": activation_dir.parent.parent.parent.parent.name,
        }
        activations_dirs.append(record)

    return activations_dirs


def load_qa_answers(activations_dirs: list[dict]) -> list[dict]:
    """Load model answers from activation directories.

    Args:
        activations_dirs: List of dictionaries containing activation directory info

    Returns:
        List of dictionaries containing model answers and metadata
    """
    model_answers = []

    for activation_dir in activations_dirs:
        answers_path = activation_dir["path"].parent / "answers.json"

        if not answers_path.exists():
            continue

        with open(answers_path) as f:
            answers = json.load(f)

        model_answers.append(activation_dir | {"answers": answers})

    return model_answers


def load_qa_results(activations_dirs: list[dict]) -> list[dict]:
    """Load model results from activation directories.

    Args:
        activations_dirs: List of dictionaries containing activation directory info

    Returns:
        List of dictionaries containing model answers and metadata
    """
    results = []

    for activation_dir in activations_dirs:
        answers_path = activation_dir["path"].parent / "answers.json"
        if not answers_path.exists():
            continue
        with open(answers_path) as f:
            answers = json.load(f)

        qa_metrics_path = activation_dir["path"].parent / "qa_metrics.json"
        if not qa_metrics_path.exists():
            continue
        with open(qa_metrics_path) as f:
            qa_metrics = json.load(f)

        record = activation_dir | {"answers": answers, "qa_metrics": qa_metrics}
        results.append(record)

    return results


def compute_qa_results(
    model_answers: list[dict],
    rouge_threshold: float = 0.3,
    pred_key: str = "prediction",
    gold_key: str = "gold",
) -> list[dict]:
    """Compute metrics for model answers.

    Args:
        model_answers: List of dictionaries containing model answers
        rouge_threshold: Threshold for considering an answer correct based on ROUGE-L

    Returns:
        List of dictionaries containing computed metrics and metadata
    """
    model_results = []

    for answer_data in model_answers:
        preds = [answer[pred_key] for answer in answer_data["answers"]]
        golds = [answer[gold_key] for answer in answer_data["answers"]]

        squad_results = compute_squad_metrics(preds, golds, return_reduced=True, return_all=True)
        rouge_results = compute_rouge_score(preds, golds, return_reduced=True, return_all=True)

        results = {
            "reduced": {**squad_results["reduced"], **rouge_results["reduced"]},
            "all": [sq | rg for sq, rg in zip(squad_results["all"], rouge_results["all"])],
        }
        results["correct"] = [
            1 if x["rougeL_fmeasure"] >= rouge_threshold else 0 for x in results["all"]
        ]

        model_results.append(answer_data | {"qa_metrics": results})

    return model_results


def load_hallucination_metrics_as_df(
    model_results: list[dict], metric_type: str = "erank"
) -> pd.DataFrame:
    """Load model metrics and combine with results into a single dataframe.

    Args:
        model_results: List of model results containing paths and metadata
        metric_type: Type of metric to load ("erank", "diff_erank", or "diff_erank_pretrain")

    Returns:
        DataFrame containing model results and metrics, with one row per sample
    """
    if not model_results:
        return pd.DataFrame()

    valid_metric_types = {"erank", "diff_erank", "diff_erank_pretrain", "logdet"}

    if metric_type not in valid_metric_types:
        raise ValueError(f"Unknown metric type: {metric_type}. Must be one of {valid_metric_types}")

    dfs = []
    for results in model_results:
        # Get appropriate metric filepath based on type
        filepath = results["path"].parent / f"{metric_type}.pt"
        if not filepath.exists():
            continue

        # Create dataframe from results
        results_df = pd.DataFrame(results["results_all"])

        # Load and process metric data
        metric_data = torch.load(filepath, weights_only=True).T
        metric_cols = [f"{metric_type}_layer_{i}" for i in range(metric_data.shape[1])]
        metric_df = pd.DataFrame(metric_data, columns=metric_cols)

        # Combine results and metrics
        df = pd.concat([results_df, metric_df], axis=1)

        # Add metadata columns
        metadata_cols = [k for k in results.keys() if k not in ["results_all", "results_reduced"]]
        for col in metadata_cols:
            df[col] = results[col]

        dfs.append(df)

    # assert len(dfs) > 0, "No valid dataframes found"
    return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()
