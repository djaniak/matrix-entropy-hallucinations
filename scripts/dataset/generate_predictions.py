import json
from pathlib import Path

import torch
import typer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from hallucinations.llm.predict import predict_with_llm
from hallucinations.metrics import compute_squad_metrics
from hallucinations.utils import sort_dataset_by_input_length

PROMPT = """
Answer the following questions as briefly as possible.
Question: What is the capital of France?
Answer: Paris

Question: Who wrote *Romeo and Juliet*?
Answer: William Shakespeare

Question: What is the boiling point of water in Celsius?
Answer: 100°C

Question: How many continents are there on Earth?
Answer: Seven

Question: What is the fastest land animal?
Answer: Cheetah

Question: {query}
Answer:
"""
BATCH_SIZE = 64


def main(
    split: str = typer.Option("validation"),
    save_dir: Path = typer.Option(...),
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    generation_kwargs = dict(
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        # return_dict_in_generate=True,
    )

    dataset = load_dataset("google-research-datasets/nq_open")
    #
    # for name, ds in dataset.items():
    #     dataset[name] = ds.select(range(10))

    dataset["validation"], reverse_sort_idx = sort_dataset_by_input_length(
        dataset["validation"], "question"
    )
    message_ds = dataset.map(
        _format_dataset,
        batched=False,
        desc="Formatting",
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    message_ds.set_transform(Encoder(tokenizer))

    outs = predict_with_llm(
        model,
        tokenizer,
        message_ds["validation"],
        batch_size=64,
        num_proc=4,
        **generation_kwargs,
    )

    golds = [ans for ans in dataset["validation"]["answer"]]
    results = [{"answer": ans, "gold": g} for ans, g in zip(outs, golds, strict=True)]
    results = [results[i] for i in reverse_sort_idx]

    metrics = compute_squad_metrics(outs, golds, return_reduced=True, return_all=True)

    with (save_dir / f"{split}_metrics.json").open("w") as file:
        json.dump(metrics, file, indent="\t")

    with (save_dir / f"{split}_outputs.json").open("w") as file:
        json.dump(results, file, indent="\t")


def _format_dataset(question: str) -> dict[str, list[dict[str, str]]]:
    return {"messages": [{"role": "user", "content": PROMPT.format(query=question)}]}


class Encoder:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, messages: dict[str, list[str]]) -> dict[str, torch.Tensor]:
        chat_input = self.tokenizer.apply_chat_template(
            messages["messages"], add_generation_prompt=True, tokenize=False
        )
        return self.tokenizer(chat_input, return_tensors="pt", padding="longest")


if __name__ == "__main__":
    typer.run(main)
