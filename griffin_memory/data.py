"""Stable document lanes; no shuffled chunks and no cross-document memory leakage."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from .tokenizer import PAD


def dataset_manifest(root, verify=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version") != 1 or manifest.get("dtype") != "<u4":
        raise ValueError("Unsupported dataset format")
    if verify:
        for filename, expected in manifest["sha256"].items():
            with (root / filename).open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Dataset integrity check failed: {filename}")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest, fingerprint


class DocumentDataset:
    def __init__(self, root, split):
        root = Path(root)
        self.offsets = np.load(root / f"{split}.offsets.npy", mmap_mode="r", allow_pickle=False)
        self.tokens = (np.memmap(root / f"{split}.bin", dtype="<u4", mode="r")
                       if self.offsets[-1] else np.empty(0, dtype="<u4"))
        self.metadata = [json.loads(line) for line in (root / f"{split}.metadata.jsonl").read_text().splitlines()]
        if len(self.metadata) != len(self.offsets) - 1:
            raise ValueError("Mismatched document offsets and metadata")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.tokens) or np.any(np.diff(self.offsets) < 2):
            raise ValueError("Invalid token offsets")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.tokens[self.offsets[index]:self.offsets[index + 1]]


@dataclass
class Cursor:
    epoch: int = 0
    group: int = 0
    chunk: int = 0

    def state_dict(self):
        return asdict(self)


class DocumentStream:
    def __init__(self, dataset, batch_size, chunk_size, seed, epochs=1, shuffle=True, cursor=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.seed = seed
        self.epochs = epochs
        self.shuffle = shuffle
        self.cursor = cursor or Cursor()
        self._epoch = None
        self._order = []

    def next(self):
        cursor = self.cursor
        if cursor.epoch >= self.epochs or not len(self.dataset):
            return None
        if self._epoch != cursor.epoch:
            self._order = list(range(len(self.dataset)))
            if self.shuffle:
                random.Random(self.seed + cursor.epoch).shuffle(self._order)
            self._epoch = cursor.epoch
        start = cursor.group * self.batch_size
        ids = self._order[start:start + self.batch_size]
        if not ids:
            raise ValueError("Invalid document cursor")
        docs = [self.dataset[i] for i in ids]
        offset = cursor.chunk * self.chunk_size
        remaining = max(len(doc) - 1 for doc in docs) - offset
        t = min(self.chunk_size, remaining)
        if t <= 0:
            raise ValueError("Invalid chunk cursor")
        x = torch.full((len(docs), t), PAD, dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, doc in enumerate(docs):
            count = max(0, min(t, len(doc) - 1 - offset))
            if count:
                x[i, :count] = torch.from_numpy(doc[offset:offset + count].astype(np.int64))
                y[i, :count] = torch.from_numpy(doc[offset + 1:offset + count + 1].astype(np.int64))
        reset = cursor.chunk == 0
        cursor.chunk += 1
        if remaining <= self.chunk_size:
            cursor.chunk = 0
            cursor.group += 1
            if cursor.group * self.batch_size >= len(self.dataset):
                cursor.group = 0
                cursor.epoch += 1
        records = [self.dataset.metadata[i] for i in ids]
        return {"tokens": x, "targets": y, "reset": reset,
                "document_ids": [f"{i}:{record['id']}" for i, record in zip(ids, records)],
                "metadata": records}
