from typing import Any

from datasets import Dataset, concatenate_datasets

from hallucinations.config import CcPromptConfig


def prepare_common_claim_labels(dataset: Dataset, label_mode: str, seed: int = 42) -> Dataset:
    # Filter out the statements without agreement
    dataset = dataset.filter(lambda x: x["agreement"] == 1.0).remove_columns("agreement")

    label_modes = {
        "binary": ["True", "False"],
        "binary_with_neither": ["True or False", "Neither"],
        "multi": ["True", "False", "Neither"],
    }
    if label_mode not in label_modes:
        raise ValueError(f"Unknown label mode: {label_mode}")

    # Adjust labels if necessary for binary_with_neither
    if label_mode == "binary_with_neither":
        dataset = dataset.map(
            lambda x: {"label": "True or False" if x["label"] in ["True", "False"] else "Neither"}
        )

    # Compute the minimum count for each label to balance the dataset
    min_n = min(
        dataset.filter(lambda x: x["label"] == label).num_rows for label in label_modes[label_mode]
    )

    # Select equal samples for each label
    balanced_dataset = concatenate_datasets(
        [
            dataset.filter(lambda x: x["label"] == label).select(range(min_n))
            for label in label_modes[label_mode]
        ]
    ).shuffle(seed=seed)

    return balanced_dataset


class CommonClaimFormatter:
    def __init__(self, prompt: CcPromptConfig):
        self.prompt = prompt

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        if self.prompt.use_prompt:
            content = self.prompt.content.format(
                **{self.prompt.statement_key: item[self.prompt.statement_key]}
            )
        else:  # Use the statement as the prompt when use_prompt is False
            content = item[self.prompt.statement_key]

        messages = {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        }

        return messages
