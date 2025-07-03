from typing import Any

from pydantic import BaseModel


class InferenceInput(BaseModel):  # type: ignore [misc]
    prompt: str
    system_prompt: str
    meta_data: dict[str, Any]


class InferenctOutput(BaseModel):  # type: ignore [misc]
    response: str
    meta_data: dict[str, Any]
