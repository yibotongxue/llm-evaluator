from typing import Any

from ...inference import InferenceFactory, InferenceInterface
from ...prompts import PromptBuilderRegistry
from ...prompts.matcher import MatcherPromptBuilder
from ...utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseJudgment
from .registry import JudgmentRegistry


@JudgmentRegistry.register("LLMMatch")
class LLMMatchJudgment(BaseJudgment):
    def __init__(self, judgment_cfgs: dict[str, Any]):
        super().__init__(judgment_cfgs=judgment_cfgs)
        self.model_cfgs: dict[str, Any] = self.judgment_cfgs.get("model_cfgs")  # type: ignore [assignment]
        self.inference_cfgs: dict[str, Any] = self.judgment_cfgs.get("inference_cfgs")  # type: ignore [assignment]
        self.cache_cfgs: dict[str, Any] | None = self.judgment_cfgs.get(
            "cache_cfgs", None
        )
        self.inference: InferenceInterface | None = None
        self.prompt_builder_type = self.judgment_cfgs["prompt_template"]
        PromptBuilderRegistry.verify_type(
            self.prompt_builder_type, MatcherPromptBuilder  # type: ignore [type-abstract]
        )

    def judge(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[bool, dict[str, Any]]]:
        ref_answers = [InferenceInput(**output.input).ref_answer for output in outputs]
        for ref_answer in ref_answers:
            if ref_answer is None:
                raise ValueError("Input does not contain a reference answer.")
        extracted_answers: list[str | None] = []
        for output in outputs:
            if output.parsed_output is not None and not isinstance(
                output.parsed_output, str
            ):
                raise ValueError(
                    f"Parsed output of judgment must be str or None, but got {output.parsed_output}."
                )
            extracted_answers.append(output.parsed_output)
        exact_matches_idx = [
            i
            for i in range(len(extracted_answers))
            if extracted_answers[i] == ref_answers[i]
        ]
        not_exact_matches_idxs = [
            i
            for i in range(len(extracted_answers))
            if extracted_answers[i] != ref_answers[i]
        ]
        if len(exact_matches_idx) == len(extracted_answers):
            return [
                (
                    True,
                    {
                        "match_reason": "exact match",
                        "extracted_answer": output.parsed_output,
                        "raw_output": output.response,
                    },
                )
                for output in outputs
            ]
        results: list[tuple[bool, dict[str, Any]]] = []
        none_idxs = [i for i in not_exact_matches_idxs if extracted_answers[i] is None]
        if len(none_idxs) == len(not_exact_matches_idxs):
            for i in range(len(outputs)):
                if i in exact_matches_idx:
                    results.append(
                        (
                            True,
                            {
                                "match_reason": "exact match",
                                "extracted_answer": outputs[i].parsed_output,
                                "ref_answer": ref_answers[i],
                                "raw_output": outputs[i].response,
                            },
                        )
                    )
                else:
                    results.append(
                        (
                            False,
                            {
                                "dismatch_reason": "extracted answer is null",
                                "extracted_answer": None,
                                "ref_answer": ref_answers[i],
                                "raw_output": outputs[i].response,
                            },
                        )
                    )
            if len(results) != len(outputs):
                raise ValueError(
                    "The number of results does not match the number of outputs."
                )
            return results
        need_judgment_indices = [
            i for i in not_exact_matches_idxs if i not in none_idxs
        ]
        inputs = [
            InferenceInput.from_prompts(prompt=extracted_answers[i]).with_ref_answer(  # type: ignore [arg-type]
                ref_answers[i]  # type: ignore [arg-type]
            )
            for i in need_judgment_indices
        ]

        if self.inference is None:
            self.inference = InferenceFactory.get_inference_instance(
                model_cfgs=self.model_cfgs,
                inference_cfgs=self.inference_cfgs,
                cache_cfgs=self.cache_cfgs,
            )
        judgments = self.inference.generate(
            inputs=inputs,
            repeat_cnt=1,
            prompt_template=self.prompt_builder_type,
            enable_tqdm=True,
            tqdm_args={"desc": "Judging outputs"},
        )
        judgment_idx = 0
        for i in range(len(outputs)):
            if i in exact_matches_idx:
                results.append(
                    (
                        True,
                        {
                            "match_reason": "exact match",
                            "extracted_answer": outputs[i].parsed_output,
                            "ref_answer": ref_answers[i],
                            "raw_output": outputs[i].response,
                        },
                    )
                )
            elif i in none_idxs:
                results.append(
                    (
                        False,
                        {
                            "dismatch_reason": "extracted answer is null",
                            "extracted_answer": None,
                            "ref_answer": ref_answers[i],
                            "raw_output": outputs[i].response,
                        },
                    )
                )
            else:
                parsed_output = judgments[judgment_idx][0].parsed_output or False
                if not isinstance(parsed_output, bool):
                    raise ValueError(
                        f"Parsed output of judgment must be bool, but got {parsed_output}."
                    )
                results.append(
                    (
                        parsed_output,
                        {
                            "extracted_answer": outputs[i].parsed_output,
                            "ref_answer": ref_answers[i],
                            "raw_output": outputs[i].response,
                            "judgment_output": judgments[judgment_idx][0].response,
                        },
                    )
                )
                judgment_idx += 1
        return results
