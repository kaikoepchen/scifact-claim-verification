"""LLM-based explanation generator using Gemma or other causal LMs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .citation import CitationContext
from .generator import Explanation

DEFAULT_MODEL = "google/gemma-4-E4B"

SYSTEM_PROMPT = (
    "You are a scientific claim verification assistant. "
    "Given a claim, a verdict, and numbered evidence sentences, write a concise "
    "2-4 sentence explanation justifying the verdict. "
    "Cite evidence using [N] markers. Only cite evidence that directly supports your reasoning. "
    "Do not repeat the claim verbatim. Be precise and scientific in tone."
)

USER_TEMPLATE = """\
Claim: {claim}
Verdict: {verdict}

Evidence:
{evidence_block}

Write a concise explanation (2-4 sentences) citing the evidence with [N] markers."""


@dataclass
class LLMGeneratorConfig:
    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    device: str | None = None


class LLMExplanationGenerator:
    """Generates cited explanations using a causal language model."""

    def __init__(self, config: LLMGeneratorConfig | None = None):
        self.config = config or LLMGeneratorConfig()
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        device = self.config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            self._model = self._model.to(device)

    def generate(self, ctx: CitationContext) -> Explanation:
        """Generate a cited explanation from a CitationContext."""
        if ctx.verdict == "NOT_ENOUGH_INFO" or not ctx.evidence:
            return Explanation(
                text="There is not enough evidence to verify this claim.",
                cited_refs=[],
                verdict=ctx.verdict,
                method="llm",
            )

        self._load_model()

        user_msg = USER_TEMPLATE.format(
            claim=ctx.claim,
            verdict=ctx.verdict,
            evidence_block=ctx.format_evidence_block(),
        )

        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_msg}"},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )

        # Decode only the generated tokens (skip the prompt)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        cited_refs = _extract_ref_ids(text)
        available_refs = set(ctx.get_ref_ids())
        cited_refs = [r for r in cited_refs if r in available_refs]

        return Explanation(
            text=text,
            cited_refs=cited_refs,
            verdict=ctx.verdict,
            method="llm",
        )

    def batch_generate(self, contexts: list[CitationContext]) -> list[Explanation]:
        return [self.generate(ctx) for ctx in contexts]


def _extract_ref_ids(text: str) -> list[int]:
    """Extract [N] citation markers from generated text."""
    return [int(m) for m in re.findall(r'\[(\d+)\]', text)]
