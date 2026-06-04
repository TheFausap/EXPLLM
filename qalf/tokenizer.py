"""Small deterministic tokenizer for the QALF proof-of-concept."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


TOKEN_PATTERN = re.compile(r"<[^>]+>|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]


@dataclass
class EncodedExample:
    prompt: list[int]
    reply: list[int]
    sequence: list[int]


class QuantumTokenizer:
    """Regex tokenizer with a fixed vocabulary and simple detokenization."""

    def __init__(self, vocab: list[str]):
        seen = set()
        ordered: list[str] = []
        for token in SPECIAL_TOKENS + vocab:
            if token not in seen:
                ordered.append(token)
                seen.add(token)
        self.vocab = ordered
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.pad_id = self.token_to_id["<pad>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.user_id = self.token_to_id["<user>"]
        self.assistant_id = self.token_to_id["<assistant>"]

    @classmethod
    def build(cls, texts: Iterable[str], vocab_size: int = 4096, min_count: int = 1) -> "QuantumTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize_text(text))
        capacity = max(vocab_size - len(SPECIAL_TOKENS), 0)
        vocab = [
            token
            for token, count in counts.most_common()
            if count >= min_count and token not in SPECIAL_TOKENS
        ][:capacity]
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokenize_text(text)]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        tokens: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.vocab):
                token = "<unk>"
            else:
                token = self.vocab[idx]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return detokenize(tokens)

    def encode_example(self, prompt: str, reply: str) -> EncodedExample:
        prompt_ids = self.encode(prompt)
        reply_ids = self.encode(reply)
        sequence = [
            self.bos_id,
            self.user_id,
            *prompt_ids,
            self.assistant_id,
            *reply_ids,
            self.eos_id,
        ]
        return EncodedExample(prompt=prompt_ids, reply=reply_ids, sequence=sequence)


def tokenize_text(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def detokenize(tokens: Iterable[str]) -> str:
    text = ""
    no_space_before = {".", ",", "!", "?", ";", ":", ")", "]", "}"}
    no_space_after = {"(", "[", "{"}
    quote_open = True
    for token in tokens:
        if not text:
            text = token
            continue
        if token == '"':
            if quote_open:
                text += ' "'
            else:
                text += '"'
            quote_open = not quote_open
        elif token in no_space_before:
            text += token
        elif text[-1] in no_space_after:
            text += token
        else:
            text += " " + token
    if text:
        text = text[0].upper() + text[1:]
    return text
