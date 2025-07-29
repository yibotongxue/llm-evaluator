from typing import Any

from ...inference import InferenceFactory, InferenceInterface
from ...prompts.matcher import MatcherPromptBuilderRegistry
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
        prompt_builder_type = self.judgment_cfgs["prompt_template"]
        self.prompt_builder = MatcherPromptBuilderRegistry.get_by_name(
            prompt_builder_type
        )()

    def judge(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[bool, dict[str, Any]]]:
        ref_answers = [InferenceInput(**output.input).ref_answer for output in outputs]
        for ref_answer in ref_answers:
            if ref_answer is None:
                raise ValueError("Input does not contain a reference answer.")
        extracted_answers = [output.extracted_answer for output in outputs]
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
                        "extracted_answer": output.extracted_answer,
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
                                "extracted_answer": outputs[i].extracted_answer,
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
        prompts = [
            self.prompt_builder.build_prompt(ref_answers[i], extracted_answers[i])  # type: ignore [arg-type]
            for i in need_judgment_indices
        ]
        inputs = [InferenceInput.from_prompts(prompt) for prompt in prompts]

        if self.inference is None:
            self.inference = InferenceFactory.get_inference_instance(
                model_cfgs=self.model_cfgs,
                inference_cfgs=self.inference_cfgs,
                cache_cfgs=self.cache_cfgs,
            )
        judgments = self.inference.generate(
            inputs=inputs,
            repeat_cnt=1,
            prompt_template=None,
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
                            "extracted_answer": outputs[i].extracted_answer,
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
                results.append(
                    (
                        self.prompt_builder.extract_answer(
                            judgments[judgment_idx][0].response
                        ),
                        {
                            "extracted_answer": outputs[i].extracted_answer,
                            "ref_answer": ref_answers[i],
                            "raw_output": outputs[i].response,
                            "prompt": prompts[judgment_idx],
                            "judgment_output": judgments[judgment_idx][0].response,
                        },
                    )
                )
                judgment_idx += 1
        return results
