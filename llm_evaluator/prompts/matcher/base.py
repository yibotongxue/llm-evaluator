from abc import ABC, abstractmethod


class BaseMatcherJudgment(ABC):
    @abstractmethod
    def build_prompt(self, ref_answer: str, answer: str) -> str:
        pass

    @abstractmethod
    def extract_answer(self, raw_output: str) -> bool:
        pass
