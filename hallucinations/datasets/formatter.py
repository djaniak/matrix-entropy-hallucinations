from typing import Any

from hallucinations.config import CcPromptConfig, PromptConfig, QaPromptConfig


class DatasetFormatter:
    def __init__(self, prompt: PromptConfig):
        self.prompt = prompt

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class NqOpenFormatter(DatasetFormatter):
    def __init__(self, prompt: QaPromptConfig, use_output: bool):
        self.prompt: QaPromptConfig = prompt
        assert self.prompt.context_key is None
        self.use_output = use_output

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        content = self.prompt.content.format(
            **{self.prompt.question_key: item[self.prompt.question_key]}
        )
        messages = {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        }
        if self.use_output:
            raise NotImplementedError("Need to determine which answer to use")

        return messages


class CommonClaimFormatter(DatasetFormatter):
    def __init__(self, prompt: CcPromptConfig):
        self.prompt: CcPromptConfig = prompt

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        if self.prompt.use_prompt:
            content = self.prompt.content.format(
                **{self.prompt.statement_key: item[self.prompt.statement_key]}
            )
        else:  # Use the statement as the prompt when use_prompt is False
            content = item[self.prompt.statement_key]

        messages = {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        }

        return messages
