from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from hallucinations.utils import load_and_resolve_config
from hallucinations.utils.misc import import_cls_from_str


class LlmConfig(BaseModel, extra="forbid"):
    name: str | Path
    tokenizer_name: str | Path
    tokenizer_padding_side: Literal["left", "right"]
    context_size: int
    compile: bool
    torch_dtype: str
    attn_implementation: str
    quantization: dict[str, Any] | None = None


class DatasetConfig(BaseModel, extra="forbid"):
    cls_path: str
    name: str | Path
    test_split_name: str
    max_answer_tokens: int
    target_column_name: str


class CsvDatasetConfig(DatasetConfig):
    dataset_url: str | None
    local_dataset_path: Path


class QaDatasetConfig(DatasetConfig, extra="forbid"):
    pass


class MMLUDatasetConfig(DatasetConfig, extra="forbid"):
    subset: str
    shuffle: bool
    subsample_ratio: float | None


class PromptConfig(BaseModel, extra="forbid"):
    content: str


class QaPromptConfig(PromptConfig, extra="forbid"):
    question_key: str
    context_key: str | None = None


class CcPromptConfig(PromptConfig, extra="forbid"):
    statement_key: str
    label_mode: Literal["binary", "binary_with_neither", "multi"]
    use_prompt: bool


class MMLUPromptConfig(PromptConfig, extra="forbid"):
    subject_key: str
    question_key: str
    choices_key: str


class GenerateActivationsConfig(BaseModel, extra="forbid"):
    llm: LlmConfig
    dataset: CsvDatasetConfig | MMLUDatasetConfig | QaDatasetConfig
    prompt: CcPromptConfig | MMLUPromptConfig | QaPromptConfig
    split: str
    batch_size: int
    generation_config: dict[str, Any]
    results_dir: Path
    random_seed: int

    @model_validator(mode="before")
    @classmethod
    def create_dataset_class(cls, data: Any) -> Any:
        dataset_config_cls_path = data["dataset"].get("cls_path")
        if dataset_config_cls_path is None:
            raise ValueError("'dataset_type' must be provided.")

        dataset_config_cls = import_cls_from_str(dataset_config_cls_path)
        data["dataset"] = dataset_config_cls(**data["dataset"])
        return data

    @property
    def max_input_length(self) -> int:
        return self.llm.context_size - self.dataset.max_answer_tokens

    @property
    def answers_file(self) -> Path:
        return self.results_dir / "answers.json"

    @property
    def metrics_file(self) -> Path:
        return self.results_dir / "metrics.json"

    @property
    def config_file(self) -> Path:
        return self.results_dir / "config.yaml"

    @property
    def activations_dir(self) -> Path:
        return self.results_dir / "activations"


class LlmJudgePromptConfig(BaseModel, extra="forbid"):
    system_prompt: str
    content: str
    question_key: str
    predicted_answer_key: str
    gold_answer_key: str

    def format(self, question: str, pred_answer: str, gold_answer: list[str]) -> str:
        return self.content.format(
            **{
                self.question_key: question,
                self.predicted_answer_key: pred_answer,
                self.gold_answer_key: str(gold_answer),
            }
        )


class LllmJudgeConfig(BaseModel, extra="forbid"):
    base_url: str | None
    llm_name: str
    prompt: LlmJudgePromptConfig
    answers_file: Path

    @property
    def dataset(self) -> QaDatasetConfig:
        return QaDatasetConfig(**load_and_resolve_config(self.config_file)["dataset"])

    @property
    def config_file(self) -> Path:
        return self.answers_file.with_name("config.yaml")

    @property
    def evaluation_file(self) -> Path:
        return self.answers_file.with_name("llm_judge.json")

    @property
    def evaluation_config_file(self) -> Path:
        return self.answers_file.with_name("llm_judge_config.yaml")


class GenerateTrajectoriesConfig(BaseModel, extra="forbid"):
    llm: LlmConfig
    dataset: CsvDatasetConfig | MMLUDatasetConfig | QaDatasetConfig
    prompt: CcPromptConfig | MMLUPromptConfig | QaPromptConfig
    results_dir: Path
    random_seed: int
    lens: Literal["tuned", "logit"]
    topk_tokens: int
    # for parallel processing
    chunk_size: int | None = None
    num_workers: int | None = None
    save_step: int | None = None

    @property
    def max_input_length(self) -> int:
        return self.llm.context_size - self.dataset.max_answer_tokens

    @property
    def answers_file(self) -> Path:
        return self.results_dir / "answers.jsonl"

    @property
    def stats_file(self) -> Path:
        return self.results_dir / "stats.jsonl"

    @property
    def config_file(self) -> Path:
        return self.results_dir / "config.yaml"
