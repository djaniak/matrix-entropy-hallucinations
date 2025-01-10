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
from tqdm.asyncio import tqdm_asyncio

from hallucinations.config import LllmJudgeConfig
from hallucinations.datasets.factory import get_dataset
from hallucinations.utils import resolve_config, save_json, save_yaml
from hallucinations.utils.misc import list_or_single_to_list

load_dotenv(override=True)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "api_key_missing")

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@hydra.main(version_base="1.3", config_path="../../config", config_name="llm_as_judge")
def main(cfg: DictConfig) -> None:
    config = LllmJudgeConfig(**resolve_config(cfg), base_url=OPENAI_API_BASE_URL)
    logger.info(f"Config: {pformat(config.model_dump())}")

    config.evaluation_config_file.parent.mkdir(parents=True, exist_ok=True)
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

    eval_func = LlmJudgeEvaluator(
        config=config,
        batch_size=BATCH_SIZE,
    )
    eval_results = asyncio.run(eval_func(answers))

    save_json(config.evaluation_file, eval_results)
    save_yaml(config.evaluation_config_file, config.model_dump())


class LlmJudgeEvaluator:
    def __init__(
        self,
        config: LllmJudgeConfig,
        batch_size: int,
    ):
        self.config = config
        self.client = openai.AsyncClient(
            base_url=config.base_url,
            api_key=OPENAI_API_KEY,
        )
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)

    async def __call__(
        self,
        predictions: list[dict[str, Any]],
    ) -> list[str | None]:
        if self.config.prompt.separate_multi_answers:
            logger.info("LLM Judge: Separating multiple answers")
            eval_func = self.eval_pred_answer_separately
        else:
            logger.info("LLM Judge: Evaluating answer once")
            eval_func = self.eval_pred_answer_once

        eval_tasks = [eval_func(ans) for ans in predictions]
        results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating predictions")

        return results

    async def eval_pred_answer_separately(
        self,
        pred: dict[str, Any],
    ) -> str | None:
        assesments: list[str | None] = []
        pred_gold = list_or_single_to_list(pred["gold"])
        for gold_answer in pred_gold:
            res = await self.do_eval_answer(
                question=pred["question"],
                pred_answer=pred["prediction"],
                gold_answers=gold_answer,
            )
            assesments.append(res)

        return self.combine_assesments(assesments)

    async def eval_pred_answer_once(
        self,
        pred: dict[str, Any],
    ) -> str | None:
        return await self.do_eval_answer(
            question=pred["question"],
            pred_answer=pred["prediction"],
            gold_answers=pred["gold"],
        )

    async def do_eval_answer(
        self,
        question: str,
        pred_answer: str,
        gold_answers: str,
    ) -> str | None:
        prompt_str = self.config.prompt.format(
            question=question,
            pred_answer=pred_answer,
            gold_answer=gold_answers,
        )
        if self.config.prompt.system_prompt is not None:
            messages = [
                {"role": "system", "content": self.config.prompt.system_prompt},
                {"role": "user", "content": prompt_str},
            ]
        else:
            messages = [{"role": "user", "content": prompt_str}]
        async with self.semaphore:
            completion = await self.client.chat.completions.create(
                model=self.config.llm_name,
                messages=messages,  # type: ignore
                temperature=0.0,
            )
        res = completion.choices[0].message.content

        return res

    def combine_assesments(self, assesments: list[str | None]) -> str | None:
        """Combines multiple assesment results into a single result.
        Note that possible answers declared in prompt are in descending priority.

        For example:
        - if the possible answers are ["correct", "incorrect", "refuse"],
            and the assesment results are ["correct", "refuse", "incorrect"],
            the combined result would be "correct".
        - if the possible answers are ["refuse", "correct", "incorrect"],
            and the assesment results are ["correct", "incorrect", "refuse"],
            the combined result would be "refuse".
        """
        for answer in self.config.prompt.possible_answers:
            if answer in assesments:
                return answer
        return None


if __name__ == "__main__":
    main()
