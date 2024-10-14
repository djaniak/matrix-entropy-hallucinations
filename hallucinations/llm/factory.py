from dataclasses import dataclass
from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from hallucinations.config import LlmConfig

LLAMA_3_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

PHI_35_MODELS = [
    "microsoft/Phi-3.5-mini-instruct",
]


@dataclass
class ModelForGeneration:
    llm: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    generate_kwargs: dict[str, Any]


def get_llm(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    if llm_config.name in LLAMA_3_MODELS:
        return get_llama_3(llm_config, **kwargs)
    elif llm_config.name in PHI_35_MODELS:
        return get_phi_35(llm_config, **kwargs)
    else:
        raise ValueError(f"Model {llm_config.name} not supported.")


def get_llama_3(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    model, tokenizer = _get_model_and_tokenizer(llm_config, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token

    return ModelForGeneration(
        llm=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        },
    )


def get_phi_35(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    model, tokenizer = _get_model_and_tokenizer(llm_config, **kwargs)

    return ModelForGeneration(
        llm=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
    )


def _get_model_and_tokenizer(
    llm_config: LlmConfig,
    **kwargs: Any,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if llm_config.quantization_config is not None:
        kwargs["quantization_config"] = BitsAndBytesConfig(**llm_config.quantization_config)

    model = AutoModelForCausalLM.from_pretrained(
        llm_config.name,
        torch_dtype=llm_config.torch_dtype,
        attn_implementation=llm_config.attn_implementation,
        **kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(llm_config.tokenizer_name)
    tokenizer.padding_side = llm_config.tokenizer_padding_side

    return model, tokenizer
