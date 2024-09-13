from typing import Any

from hallucinations.config import QaPromptConfig


class NqOpenFormatter:
    def __init__(self, prompt: QaPromptConfig, use_output: bool):
        self.prompt = prompt
        assert self.prompt.context_key is None
        self.use_output = use_output

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        content = self.prompt.content.format(**{self.prompt.question_key: item["question"]})
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
