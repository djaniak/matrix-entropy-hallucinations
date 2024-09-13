from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from hallucinations.config import LlmConfig

LLAMA_3_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]


@dataclass
class ModelForGeneration:
    llm: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generate_kwargs: dict[str, Any]


def get_llm(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    if llm_config.name in LLAMA_3_MODELS:
        return get_llama_3(llm_config, **kwargs)
    else:
        raise ValueError(f"Model {llm_config.name} not supported.")


def get_llama_3(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    model, tokenizer = _get_model_tokenizer(llm_config, **kwargs)

    tokenizer.padding_side = llm_config.tokenizer_padding_side
    tokenizer.pad_token = tokenizer.eos_token

    return ModelForGeneration(
        llm=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        },
    )


def _get_model_tokenizer(
    llm_config: LlmConfig,
    **kwargs: Any,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if llm_config.use_bnb_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if torch.cuda.is_available():
        assert is_flash_attn_2_available()
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        llm_config.name,
        torch_dtype=torch.bfloat16,
        **kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(llm_config.tokenizer_name)
    return model, tokenizer
