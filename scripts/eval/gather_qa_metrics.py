from pathlib import Path

import numpy as np
import pandas as pd
import typer

from hallucinations.utils.defaults import (
    ACTIVATIONS_DIR,
    DATASET_NAMES_MAPPING,
    MODEL_NAMES_MAPPING,
    PROMPT_NAMES_MAPPING,
)
from hallucinations.utils.metric_loader import get_activation_dirs, load_qa_results


def main(
    output_dir: Path = typer.Option(..., help="Path to save the computed metric values"),
) -> None:
    activations_dirs = get_activation_dirs(ACTIVATIONS_DIR)
    model_results = load_qa_results(activations_dirs)

    data = []
    for result in model_results:
        model_name = MODEL_NAMES_MAPPING[result["model"]]
        prompt_name = PROMPT_NAMES_MAPPING[result["prompt"]]
        dataset_name = DATASET_NAMES_MAPPING[result["dataset"]]
        temp_name = result["temperature"].capitalize()

        rouge_l = result["qa_metrics"]["reduced"]["rougeL_fmeasure"]
        labels = [1 if x["rougeL_fmeasure"] >= 0.3 else 0 for x in result["qa_metrics"]["all"]]
        accuracy = np.mean(labels)

        data.append(
            {
                "Model": model_name,
                "Dataset": dataset_name,
                "Temperature": temp_name,
                "Prompt": prompt_name,
                "ROUGE-L": round(rouge_l, 2),
                "Accuracy": round(accuracy, 2),
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values(["Dataset", "Model", "Temperature", "Prompt"])
    print(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(output_dir / "qa_metrics.json", orient="records", indent=2)
    with open(output_dir / "qa_metrics.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: "%.2f" % x))


if __name__ == "__main__":
    typer.run(main)
