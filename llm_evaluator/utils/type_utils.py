from __future__ import annotations

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


class InferenctOutput(BaseModel):  # type: ignore [misc]
    response: str
    input: dict[str, Any]
    engine: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")
