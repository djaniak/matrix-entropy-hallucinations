import time
from typing import Any

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


def predict_with_llm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int,
    num_proc: int,
    **generate_kwargs: dict[str, Any],
) -> list[str]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_proc,
        pin_memory=(num_proc > 1),
        shuffle=False,
    )

    model_outputs = []

    device = next(model.parameters()).device

    with tqdm(dataloader) as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_length = input_ids.size(1)

            start_time = time.time()
            generated_ids = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
            duration = time.time() - start_time

            decoded = tokenizer.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )
            model_outputs.extend(decoded)

            stats = {
                "input_size": input_length,
                "throughput": f"{generated_ids.numel() / duration:0.2f} tok/sec",
                "mean(#special_tokens)": f"{(1 - attention_mask).float().mean().item():0.3f}",
            }
            pbar.set_postfix(stats)

    return model_outputs
