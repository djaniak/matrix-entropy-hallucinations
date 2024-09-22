import asyncio
import json
import logging
import os
from pathlib import Path
from pprint import pformat
from typing import Any

import hydra
import openai
import yaml
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel, Field, SecretStr
from tqdm import trange

from hallucinations.config import QaDatasetConfig
from hallucinations.datasets.factory import get_dataset
from hallucinations.utils.misc import load_and_resolve_config, resolve_config

load_dotenv()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
    base_url: str | None = Field(os.getenv("OPENAI_API_BASE_URL", None))
    api_key: SecretStr = Field(os.getenv("OPENAI_API_KEY", "api_key_missing"))
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


@hydra.main(version_base="1.3", config_path="../../config", config_name="llm_as_judge")
def main(cfg: DictConfig) -> None:
    config = LllmJudgeConfig(**resolve_config(cfg))
    logger.info(f"Config: {pformat(config.model_dump())}")

    dataset = get_dataset(config.dataset, split=config.dataset.test_split_name)

    with config.answers_file.open("r") as file:
        answers = json.load(file)

    answers = [
        {
            "question": ds_item["question"],
            "answer": ans_item["answer"],
            "gold": ans_item["gold"],
        }
        for ds_item, ans_item in zip(dataset, answers, strict=True)
    ]

    eval_func = LlmJudgeEvaluator(config)
    eval_results = asyncio.run(eval_func(answers))

    with config.evaluation_file.open("w") as file:
        json.dump(eval_results, file, indent="\t")

    with config.evaluation_config_file.open("w") as file:
        yaml.dump(config.model_dump(), file)


class LlmJudgeEvaluator:
    def __init__(self, config: LllmJudgeConfig):
        self.config = config
        self.client = openai.AsyncClient(
            base_url=config.base_url,
            api_key=str(config.api_key),
        )

    async def __call__(self, answers: list[dict[str, Any]]) -> list[str]:
        results = []
        for i in trange(0, len(answers), BATCH_SIZE, desc="Evaluating batches"):
            batch_tasks = [self.eval_single_answer(ans) for ans in answers[i : i + BATCH_SIZE]]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

        return results

    async def eval_single_answer(self, answer: dict[str, Any]) -> str:
        prompt_str = self.config.prompt.format(
            answer["question"],
            answer["answer"],
            answer["gold"],
        )
        completion = await self.client.chat.completions.create(
            model=self.config.llm_name,
            messages=[
                {"role": "system", "content": self.config.prompt.system_prompt},
                {"role": "user", "content": prompt_str},
            ],
        )
        return completion.choices[0].message.content or "<no_response>"


if __name__ == "__main__":
    main()
