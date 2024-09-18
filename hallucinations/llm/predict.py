import time
from pathlib import Path

import torch
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerateDecoderOnlyOutput, GenerationConfig


def predict_with_llm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    generation_config: GenerationConfig,
    batch_size: int,
    num_proc: int,
    activations_save_dir: Path | None = None,
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

    for i, batch in (
        pbar := tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Generating predictions",
        )
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        input_length = input_ids.size(1)

        start_time = time.time()
        generations = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        duration = time.time() - start_time

        if isinstance(generations, GenerateDecoderOnlyOutput):
            assert activations_save_dir is not None
            generated_ids = generations.sequences
            # Process and save activations from the first forward pass (no autoregressive decoding)
            # We need to stack a tuple of tensors representing the hidden states/attentions of different layers
            if hasattr(generations, "hidden_states"):
                hidden_states = torch.stack([t.cpu() for t in generations.hidden_states[0]])
                torch.save(hidden_states, activations_save_dir / f"batch_hidden_states_{i}.pt")
            if hasattr(generations, "attentions"):
                attentions = torch.stack([t.cpu() for t in generations.attentions[0]])
                torch.save(attentions, activations_save_dir / f"batch_attentions_{i}.pt")
        elif isinstance(generations, Tensor):
            generated_ids = generations
        else:
            raise ValueError(f"Unexpected generation output: {type(generations)}")

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

        del generations, input_ids, attention_mask, generated_ids
        torch.cuda.empty_cache()

    return model_outputs
