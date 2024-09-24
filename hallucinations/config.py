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


class DatasetConfig(BaseModel, extra="forbid"):
    name: str | Path
    split_name: str | None
    max_answer_tokens: int


class CsvDatasetConfig(DatasetConfig):
    dataset_url: str | None
    local_dataset_path: Path


class QaDatasetConfig(DatasetConfig, extra="forbid"):
    pass


class PromptConfig(BaseModel, extra="forbid"):
    content: str


class QaPromptConfig(PromptConfig, extra="forbid"):
    question_key: str
    context_key: str | None = None


class CcPromptConfig(PromptConfig, extra="forbid"):
    statement_key: str
    label_mode: Literal["binary", "binary_with_neither", "multi"]
    use_prompt: bool


class GenerateActivationsConfig(BaseModel, extra="forbid"):
    llm: LlmConfig
    dataset: DatasetConfig
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
