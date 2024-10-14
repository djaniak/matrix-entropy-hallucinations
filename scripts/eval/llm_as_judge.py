import asyncio
import json
import logging
import os
from pprint import pformat
from typing import Any

import hydra
import openai
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import trange

from hallucinations.config import LllmJudgeConfig
from hallucinations.datasets.factory import get_dataset
from hallucinations.utils import resolve_config, save_json, save_yaml

load_dotenv()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "api_key_missing")

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@hydra.main(version_base="1.3", config_path="../../config", config_name="llm_as_judge")
def main(cfg: DictConfig) -> None:
    config = LllmJudgeConfig(**resolve_config(cfg), base_url=OPENAI_API_BASE_URL)
    logger.info(f"Config: {pformat(config.model_dump())}")

    dataset = get_dataset(config.dataset, split=config.dataset.test_split_name)

    with config.answers_file.open("r") as file:
        answers = json.load(file)

    answers = [
        {
            "question": ds_item["question"],
            "prediction": ans_item["prediction"],
            "gold": ans_item["gold"],
        }
        for ds_item, ans_item in zip(dataset, answers, strict=True)
    ]

    eval_func = LlmJudgeEvaluator(config)
    eval_results = asyncio.run(eval_func(answers))

    save_json(config.evaluation_file, eval_results)
    save_yaml(config.evaluation_config_file, config.model_dump())


class LlmJudgeEvaluator:
    def __init__(self, config: LllmJudgeConfig):
        self.config = config
        self.client = openai.AsyncClient(
            base_url=config.base_url,
            api_key=OPENAI_API_KEY,
        )

    async def __call__(self, predictions: list[dict[str, Any]]) -> list[str]:
        results = []
        for i in trange(0, len(predictions), BATCH_SIZE, desc="Evaluating batches"):
            batch_tasks = [self.eval_single_answer(ans) for ans in predictions[i : i + BATCH_SIZE]]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

        return results

    async def eval_single_answer(self, pred: dict[str, Any]) -> str:
        prompt_str = self.config.prompt.format(
            pred["question"],
            pred["prediction"],
            pred["gold"],
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
