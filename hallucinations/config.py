from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class LlmConfig(BaseModel, extra="forbid"):
    name: str | Path
    tokenizer_name: str | Path
    tokenizer_padding_side: Literal["left", "right"]
    context_size: int
    compile: bool
    torch_dtype: str
    attn_implementation: str
    quantization_config: dict[str, Any] | None = None


class QaDatasetConfig(BaseModel, extra="forbid"):
    name: str | Path
    test_split_name: str
    max_answer_tokens: int


class QaPromptConfig(BaseModel, extra="forbid"):
    content: str
    question_key: str
    context_key: str | None = None


class GenerateActivationsConfig(BaseModel, extra="forbid"):
    llm: LlmConfig
    dataset: QaDatasetConfig
    prompt: QaPromptConfig
    batch_size: int
    generation_config: dict[str, Any]
    results_dir: Path
    random_seed: int

    @property
    def max_input_length(self) -> int:
        return self.llm.context_size - self.dataset.max_answer_tokens

    @property
    def answers_file(self) -> Path:
        return self.results_dir / "answers.jsonl"

    @property
    def metrics_file(self) -> Path:
        return self.results_dir / "metrics.json"

    @property
    def config_file(self) -> Path:
        return self.results_dir / "config.yaml"

    @property
    def activations_dir(self) -> Path:
        return self.results_dir / "activations"
