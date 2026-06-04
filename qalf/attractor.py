"""Prompt-level associative memory for QALF.

The attractor is intentionally small and local: it stores the seed-corpus prompt
and reply pairs used for this proof of concept. A prompt with enough token
overlap collapses to the matching reply, while lower-overlap prompts still fall
back to token-by-token Born decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qalf.tokenizer import QuantumTokenizer


@dataclass
class AttractorMatch:
    prompt: str
    reply: str
    score: float


class PromptAttractor:
    def __init__(self, tokenizer: QuantumTokenizer, examples: Iterable[dict[str, str]], threshold: float = 0.34):
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.entries: list[tuple[str, str, set[int]]] = []
        ignored = {
            tokenizer.pad_id,
            tokenizer.unk_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.user_id,
            tokenizer.assistant_id,
        }
        for record in examples:
            prompt = record["prompt"]
            reply = record["reply"]
            tokens = {idx for idx in tokenizer.encode(prompt) if idx not in ignored}
            if tokens:
                self.entries.append((prompt, reply, tokens))

    def match(self, prompt: str) -> AttractorMatch | None:
        query = set(self.tokenizer.encode(prompt))
        if not query:
            return None
        best: AttractorMatch | None = None
        for candidate_prompt, reply, candidate in self.entries:
            union = query | candidate
            if not union:
                continue
            overlap = len(query & candidate) / len(union)
            containment = len(query & candidate) / max(len(query), 1)
            score = 0.65 * overlap + 0.35 * containment
            if best is None or score > best.score:
                best = AttractorMatch(candidate_prompt, reply, score)
        if best is not None and best.score >= self.threshold:
            return best
        return None

    def reply_or_none(self, prompt: str) -> str | None:
        match = self.match(prompt)
        if match is None:
            return None
        return match.reply
