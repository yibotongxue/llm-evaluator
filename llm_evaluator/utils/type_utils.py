from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict


class CustomBaseModel(BaseModel):  # type: ignore [misc]
    model_config = ConfigDict(extra="allow")

    def to_brief_dict(self) -> dict[str, Any]:
        raw_dict = deepcopy(self.model_dump())
        if "meta_data" in raw_dict:
            raw_dict.pop("meta_data")
        return raw_dict  # type: ignore [no-any-return]


class InferenceInput(CustomBaseModel):
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


class InferenceOutput(CustomBaseModel):
    response: str
    input: dict[str, Any]
    engine: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")


class MetricsOutput(CustomBaseModel):
    metrics_name: str
    metrics: float
    meta_data: dict[str, Any] | list[dict[str, Any]]


class EvaluateResult(CustomBaseModel):
    metrics: list[MetricsOutput]
    benchmark_cfgs: BenchmarkConfigs
    raw_output: list[InferenceOutput]


class BenchmarkConfigs(CustomBaseModel):
    data_name_or_path: str
    data_template: str
    task_list: list[str] | None
    data_size: int | None
    metrics_cfgs: list[dict[str, Any]]

    model_config = ConfigDict(extra="allow")


class EvalConfigs(CustomBaseModel):
    benchmarks: dict[str, BenchmarkConfigs]
    attack_cfgs: list[dict[str, Any]]

    model_config = ConfigDict(extra="allow")


def to_dict(obj: BaseModel | dict[str, Any]) -> dict[str, Any]:

    def _to_dict(
        obj: BaseModel | dict[str, Any] | list[Any] | Any
    ) -> dict[str, Any] | list[Any] | Any:
        if isinstance(obj, BaseModel):
            return _to_dict(obj.model_dump())
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_dict(e) for e in obj]
        return obj

    return _to_dict(obj)  # type: ignore [return-value]


def to_breif_dict(obj: CustomBaseModel | BaseModel | dict[str, Any]) -> dict[str, Any]:

    def _to_brief_dict(
        obj: CustomBaseModel | BaseModel | dict[str, Any] | list[Any] | Any
    ) -> dict[str, Any] | list[Any] | Any:
        if isinstance(obj, CustomBaseModel):
            return _to_brief_dict(obj.to_brief_dict())
        elif isinstance(obj, BaseModel):
            return _to_brief_dict(obj.model_dump())
        if isinstance(obj, dict):
            return {
                k: _to_brief_dict(v) for k, v in obj.items() if not k == "meta_data"
            }
        if isinstance(obj, list):
            return [_to_brief_dict(e) for e in obj]
        return obj

    return _to_brief_dict(obj)  # type: ignore [return-value]
