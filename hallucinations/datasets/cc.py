from typing import Any

from datasets import Dataset, concatenate_datasets

from hallucinations.config import CcPromptConfig


def prepare_common_claim_labels(dataset: Dataset, label_mode: str, seed: int = 42) -> Dataset:
    # Filter out the statements without agreement
    dataset = dataset.filter(lambda x: x["agreement"] == 1.0)
    dataset = dataset.remove_columns("agreement")

    if label_mode == "binary":
        # Filter out the 'Neither' label
        true_false_dataset = dataset.filter(lambda x: x["label"] in ["True", "False"])
        true_false_dataset = true_false_dataset.map(
            lambda x: {"label": 1 if x["label"] == "True" else 0}
        )

        # Balance the dataset
        n_true, n_false = (
            len(true_false_dataset.filter(lambda x: x["label"] == 1)),
            len(true_false_dataset.filter(lambda x: x["label"] == 0)),
        )
        n = min([n_true, n_false])
        true_statements = true_false_dataset.filter(lambda x: x["label"] == 1).select(range(n))
        false_statements = true_false_dataset.filter(lambda x: x["label"] == 0).select(range(n))

        true_false_dataset = concatenate_datasets([true_statements, false_statements]).shuffle(
            seed=seed
        )
        return true_false_dataset

    elif label_mode == "binary_with_neither":
        # Convert labels to binary with 'neither' (1 for true/false, 0 for neither)
        binary_with_neither_dataset = dataset.map(
            lambda x: {"label": 1 if x["label"] in ["True", "False"] else 0}
        )

        # Balance the dataset between 'true/false' and 'neither'
        n_true_false = len(binary_with_neither_dataset.filter(lambda x: x["label"] == 1))
        n_neither = len(binary_with_neither_dataset.filter(lambda x: x["label"] == 0))
        n = min([n_true_false, n_neither])
        true_false_statements = binary_with_neither_dataset.filter(
            lambda x: x["label"] == 1
        ).select(range(n))
        neither_statements = binary_with_neither_dataset.filter(lambda x: x["label"] == 0).select(
            range(n)
        )

        binary_with_neither_dataset = concatenate_datasets(
            [true_false_statements, neither_statements]
        ).shuffle(seed=seed)
        return binary_with_neither_dataset

    elif label_mode == "multi":
        # Map the labels to a numerical representation: True -> 1, False -> 0, Neither -> 2
        multi_label_dataset = dataset.map(
            lambda x: {"label": 1 if x["label"] == "True" else (0 if x["label"] == "False" else 2)}
        )

        # Balance the dataset between 'true', 'false' and 'neither'
        n_true = len(multi_label_dataset.filter(lambda x: x["label"] == 1))
        n_false = len(multi_label_dataset.filter(lambda x: x["label"] == 0))
        n_neither = len(multi_label_dataset.filter(lambda x: x["label"] == 2))
        n = min([n_true, n_false, n_neither])
        true_statements = multi_label_dataset.filter(lambda x: x["label"] == 1).select(range(n))
        false_statements = multi_label_dataset.filter(lambda x: x["label"] == 0).select(range(n))
        neither_statements = multi_label_dataset.filter(lambda x: x["label"] == 2).select(range(n))

        multi_label_dataset = concatenate_datasets(
            [true_statements, false_statements, neither_statements]
        ).shuffle(seed=seed)
        return multi_label_dataset

    else:
        raise ValueError(f"Unknown label mode: {label_mode}")


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
