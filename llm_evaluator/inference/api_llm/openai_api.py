import os
from typing import Any

import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from ...utils.logger import Logger
from ...utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseApiLLMInference

__all__ = [
    "OpenAIApiLLMInference",
]


class OpenAIApiLLMInference(BaseApiLLMInference):
    _DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    _QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    _BASE_URL_MAP: dict[str, str] = {
        "deepseek-chat": _DEEPSEEK_BASE_URL,
        "deepseek_reasoner": _DEEPSEEK_BASE_URL,
        "qwen-max": _QWEN_BASE_URL,
        "qwen-plus": _QWEN_BASE_URL,
    }

    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        super().__init__(model_cfgs=model_cfgs, inference_cfgs=inference_cfgs)
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.model_name = self.model_cfgs["model_name_or_path"]
        api_key = os.environ.get(self.model_cfgs.get("api_key_name", "OPENAI_API_KEY"))
        base_url = self._BASE_URL_MAP.get(self.model_name, None)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _single_generate(self, inference_input: InferenceInput) -> InferenceOutput:
        messages: list[ChatCompletionMessageParam] = []
        messages.append(
            {
                "role": "system",
                "content": inference_input.system_prompt,
            }
        )
        for turn in inference_input.conversation:
            messages.append(
                {
                    "role": turn["role"],
                    "content": turn["content"],
                }
            )
        for i in range(self.max_retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    **self.inference_cfgs,
                )
            except Exception as err:
                self.logger.error(
                    msg=f"第{i+1}次呼叫{self.model_name} API失败，错误信息为{err}"
                )
                continue
            content = response.choices[0].message.content
            return InferenceOutput(
                response=content,
                input=inference_input.model_dump(),
                engine="api",
                meta_data=response.model_dump(),
            )
        self.logger.error(
            msg=f"所有对{self.model_name} API的呼叫均以失败，返回默认信息"
        )
        return InferenceOutput(
            response="",
            input=inference_input.model_dump(),
            engine="api",
            meta_data={},
        )
