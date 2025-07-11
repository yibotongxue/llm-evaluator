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


class InferenctOutput(BaseModel):  # type: ignore [misc]
    response: str
    input: dict[str, Any]
    engine: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")
