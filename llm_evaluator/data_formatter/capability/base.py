from ...prompts import BasePromptBuilder
from ..base import BaseDataFormatter


class BaseCapabilityDataFormatter(BaseDataFormatter):
    def __init__(self, prompt_builder: BasePromptBuilder):
        self.prompt_builder = prompt_builder
