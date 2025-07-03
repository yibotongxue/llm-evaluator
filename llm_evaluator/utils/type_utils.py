from typing import Any

from pydantic import BaseModel, ConfigDict


class InferenceInput(BaseModel):  # type: ignore [misc]
    prompt: str
    system_prompt: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")


class InferenctOutput(BaseModel):  # type: ignore [misc]
    response: str
    input: dict[str, Any]
    engine: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")
