from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import torch
from loguru import logger
from torch import Tensor
from transformers import PreTrainedModel
from transformers.generation import GenerateDecoderOnlyOutput

# hf-transformers return a tuple of tensors representing the hidden states/attentions of different layers
# dimensions are: (num_new_tokens, num_layers, tensor[batch_size, sequence_length, hidden_size])
# in particular, data for token 0 represents hidden states for the whole input sequence (thus sequence_length = input_length)
# and data for the rest of the tokens represents hidden states for the generated tokens (thus each one has sequence_length = 1)
# NOTE 1: Special tokens don't mark chat-template tokens
# NOTE 2: Even when mask chat-template tokens is given, not all might be covered, e.g., in <spec_tok> and <spec_tok> "and" won't be marked


class ActivationStorage(ABC):
    """Extract intermediate states of an LLM and save them to disk."""

    def __init__(self, activations_save_dir: Path, verbose: bool = True):
        self.activations_save_dir = activations_save_dir
        self.verbose = verbose

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self.update(*args, **kwargs)

    @abstractmethod
    def update(
        self,
        outputs: GenerateDecoderOnlyOutput,
        attention_mask: Tensor,
        special_token_mask: Tensor,
        decoder_added_token_mask: Tensor,
        input_length: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

    def flush(self) -> None:
        if self.verbose:
            files = list(self.activations_save_dir.glob("*.pt"))
            size = sum(file.stat().st_size for file in files)
            logger.info(f"Stored total {size*1e-9:0.1f}GB in {len(files)} files")


class AllActivationsStorage(ActivationStorage):
    def update(
        self,
        outputs: GenerateDecoderOnlyOutput,
        attention_mask: Tensor,
        special_token_mask: Tensor,
        decoder_added_token_mask: Tensor,
        input_length: int,
        **kwargs: Any,
    ) -> None:
        batch_idx = kwargs["batch_idx"]
        intermediate_states = {
            "attention_mask": attention_mask.cpu(),
            "special_token_mask": special_token_mask.cpu(),
            "decoder_token_mask": decoder_added_token_mask.cpu(),
            "input_length": input_length,
        }
        if "hidden_states" in outputs:
            intermediate_states["hidden_states"] = get_sequences_by_layer(
                outputs.hidden_states, concat=True
            )
        if "attentions" in outputs:
            intermediate_states["attentions"] = get_sequences_by_layer(
                outputs.attentions, concat=False
            )

        save_file = self.activations_save_dir / f"batch_{batch_idx}.pt"
        torch.save(intermediate_states, save_file)
        if self.verbose:
            logger.info(
                f"Saved ({save_file.stat().st_size*1e-9:0.1f}GB) activations to {save_file}"
            )


class MultipleSamplesActivationStorage(ActivationStorage):
    """Storage class for handling activations when generating multiple samples per input.

    This class extends ActivationStorage to handle cases where the model generates multiple output
    samples for each input sequence, such as when using different sampling temperatures. It properly
    reshapes and stores the hidden states, attention patterns, and log likelihoods for each sample.
    """

    def update(
        self,
        outputs: GenerateDecoderOnlyOutput,
        attention_mask: Tensor,
        special_token_mask: Tensor,
        decoder_added_token_mask: Tensor,
        input_length: int,
        model: PreTrainedModel,
        batch_size: int,
        num_samples: int,
        temperature: Literal["low_temperature", "high_temperature"],
        **kwargs: Any,
    ) -> None:
        batch_idx = kwargs["batch_idx"]
        intermediate_states = {
            "attention_mask": attention_mask.cpu(),
            "special_token_mask": special_token_mask.cpu(),
            "decoder_token_mask": decoder_added_token_mask.cpu(),
            "input_length": input_length,
            "num_samples": num_samples,
        }
        if "hidden_states" in outputs:
            hidden_states = get_sequences_by_layer(outputs.hidden_states, concat=True)
            intermediate_states["hidden_states"] = maybe_reshape_multiple_samples(
                x=hidden_states,
                temperature=temperature,
                data_type="hidden_states",
                batch_size=batch_size,
                num_samples=num_samples,
            )

        if "attentions" in outputs:
            attentions = get_sequences_by_layer(outputs.attentions, concat=False)
            intermediate_states["attentions"] = maybe_reshape_multiple_samples(
                x=attentions,
                temperature=temperature,
                data_type="attentions",
                batch_size=batch_size,
                num_samples=num_samples,
            )

        if "scores" in outputs:
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            intermediate_states["log_likelihoods"] = maybe_reshape_multiple_samples(
                x=transition_scores,
                temperature=temperature,
                data_type="scores",
                batch_size=batch_size,
                num_samples=num_samples,
            )

        save_file = self.activations_save_dir / f"batch_{batch_idx}_{temperature}.pt"
        torch.save(intermediate_states, save_file)
        if self.verbose:
            logger.info(
                f"Saved ({save_file.stat().st_size*1e-9:0.1f}GB) activations to {save_file}"
            )


def get_sequences_by_layer(
    interm_state: tuple[tuple[Tensor]],
    concat: bool,
) -> list[Tensor] | list[list[Tensor]]:
    layerwise_states: list[list[Tensor]] = []
    for gen_token_data in interm_state:
        for layer_idx, layer_data in enumerate(gen_token_data):
            try:
                layerwise_states[layer_idx].append(layer_data.cpu())
            except IndexError:
                layerwise_states.append([layer_data.cpu()])

    if concat:
        return [torch.cat(layer_data, dim=-2) for layer_data in layerwise_states]
    else:
        return layerwise_states


def maybe_reshape_multiple_samples(
    x: Tensor | list[Tensor] | list[list[Tensor]],
    temperature: Literal["low_temperature", "high_temperature"],
    data_type: Literal["hidden_states", "attentions", "scores"],
    batch_size: int,
    num_samples: int,
) -> Tensor | list[Tensor] | list[list[Tensor]]:
    if temperature == "high_temperature":
        if data_type == "hidden_states":
            x = [hs.reshape(batch_size, num_samples, *hs.shape[1:]) for hs in x]  # type: ignore
        elif data_type == "attentions":
            for i, attn in enumerate(x):
                for j, layer in enumerate(attn):
                    x[i][j] = layer.reshape(batch_size, num_samples, *layer.shape[1:])  # type: ignore
        elif data_type == "scores":
            x = x.reshape(batch_size, num_samples, -1)  # type: ignore
        else:
            raise ValueError(f"Unknown type: {data_type}")
    return x
