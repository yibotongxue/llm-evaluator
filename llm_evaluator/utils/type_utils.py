from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict


class InferenceInput(BaseModel):  # type: ignore [misc]
    conversation: list[dict[str, Any]]
    prefilled: bool
    system_prompt: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_prompts(
        cls: type[InferenceInput], prompt: str, system_prompt: str = ""
    ) -> InferenceInput:
        return cls(
            conversation=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            prefilled=False,
            system_prompt=system_prompt,
            meta_data={},
        )

    def get_raw_question(self) -> str:
        if "raw_question" in self.meta_data:
            return self.meta_data["raw_question"]  # type: ignore [no-any-return]
        if self.prefilled:
            return self.conversation[-2]["content"]  # type: ignore [no-any-return]
        return self.conversation[-1]["content"]  # type: ignore [no-any-return]

    def with_system_prompt(self, system_prompt: str) -> InferenceInput:
        raw = {
            **self.model_dump(),
            "system_prompt": system_prompt,
        }
        return InferenceInput(**raw)

    def with_meta_data(self, meta_data: dict[str, Any]) -> InferenceInput:
        new_meta_data = {
            **self.meta_data,
            **meta_data,
        }
        raw = {
            **self.model_dump(),
            "meta_data": new_meta_data,
        }
        return InferenceInput(**raw)

    def with_prefill(self, prefix: str) -> InferenceInput:
        new_conversation = deepcopy(self.conversation)
        last_message = new_conversation[-1]
        if last_message["role"] == "assistant":
            last_message["content"] = prefix
        else:
            new_conversation.append({"role": "assistant", "content": prefix})
        raw = {
            **self.model_dump(),
            "conversation": new_conversation,
        }
        return InferenceInput(**raw)


class InferenceOutput(BaseModel):  # type: ignore [misc]
    response: str
    input: dict[str, Any]
    engine: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")


class EvaluateResult(BaseModel):  # type: ignore [misc]
    metrics: dict[str, float]
    benchmark_cfgs: BenchmarkConfigs
    raw_output: list[InferenceOutput]


class BenchmarkConfigs(BaseModel):  # type: ignore [misc]
    data_name_or_path: str
    data_template: str
    task_list: list[str] | None

    model_config = ConfigDict(extra="allow")


class EvalConfigs(BaseModel):  # type: ignore [misc]
    benchmarks: dict[str, BenchmarkConfigs]
    attack_cfgs: list[dict[str, Any]]
    metrics_cfgs: list[dict[str, Any]]

    model_config = ConfigDict(extra="allow")
