from statistics import mean

from torchmetrics.functional.text import rouge_score
from tqdm import tqdm

ROUGE_KEYS = "rougeL"


def compute_rouge_score(
    preds: list[str],
    golds: list[str | list[str]],
    return_all: bool,
    return_reduced: bool,
    rouge_keys: str | tuple[str] = ROUGE_KEYS,
) -> dict[str, dict[str, float] | list[dict[str, float]]]:
    assert return_reduced or return_all
    scores = []
    for pred, ans in tqdm(
        zip(preds, golds, strict=True),
        desc="Evaluating ROUGE metrics",
        total=len(preds),
    ):
        item_scores = rouge_score(preds=pred, target=ans, rouge_keys=rouge_keys)
        scores.append({key: val.item() for key, val in item_scores.items()})

    results: dict[str, dict[str, float] | list[dict[str, float]]] = {}
    if return_reduced:
        # computing average as in https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/text/rouge.py#L393
        reduced_scores = {key: mean([item[key] for item in scores]) for key in scores[0].keys()}
        results["reduced"] = reduced_scores

    if return_all:
        results["all"] = scores

    return results
