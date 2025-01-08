from dataclasses import dataclass
from typing import Any, Callable

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from hallucinations.config import LlmConfig

MODEL_REGISTRY: dict[str, Callable[[LlmConfig, Any], "ModelForGeneration"]] = {
    "meta-llama": lambda config, kwargs: _get_model_for_generation(config, kwargs, True),
    "microsoft": lambda config, kwargs: _get_model_for_generation(config, kwargs, False),
    "mistralai": lambda config, kwargs: _get_model_for_generation(config, kwargs, False),
    "Qwen": lambda config, kwargs: _get_model_for_generation(config, kwargs, False),
}


@dataclass
class ModelForGeneration:
    llm: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    generate_kwargs: dict[str, Any]


def get_llm(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    for model_prefix, handler in MODEL_REGISTRY.items():
        if llm_config.name.startswith(model_prefix):
            return handler(llm_config, kwargs)
    raise ValueError(f"Model {llm_config.name} not supported.")


def _get_model_for_generation(
    llm_config: LlmConfig, kwargs: Any, set_pad_to_eos: bool
) -> ModelForGeneration:
    model, tokenizer = _get_model_and_tokenizer(llm_config, **kwargs)
    if set_pad_to_eos:
        tokenizer.pad_token = tokenizer.eos_token

    return ModelForGeneration(
        llm=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id if set_pad_to_eos else tokenizer.pad_token_id,
        },
    )


def _get_model_and_tokenizer(
    llm_config: LlmConfig,
    **kwargs: Any,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if llm_config.quantization is not None:
        kwargs["quantization_config"] = BitsAndBytesConfig(**llm_config.quantization)

    use_untrained = getattr(llm_config, "untrained", False)

    if use_untrained:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = AutoConfig.from_pretrained(llm_config.name)
        model = AutoModelForCausalLM.from_config(config).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            llm_config.name,
            torch_dtype=llm_config.torch_dtype,
            attn_implementation=llm_config.attn_implementation,
            **kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(llm_config.tokenizer_name)
    tokenizer.padding_side = llm_config.tokenizer_padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
