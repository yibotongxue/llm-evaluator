from typing import Any

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .base import BaseInference


class HuggingFaceInference(BaseInference):
    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        super().__init__(model_cfgs=model_cfgs, inference_cfgs=inference_cfgs)
        model_name = self.model_cfgs["model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name
        )
        self.accelerator = Accelerator()
        self.model: PreTrainedModel = self.accelerator.prepare(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.apply_chat_template()
