"""Structured job logging for train/eval runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JobLogger:
    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None
        self.handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a", encoding="utf-8")

    def emit(self, record: dict[str, Any]) -> None:
        enriched = {"time": time.time(), **record}
        line = json.dumps(enriched, ensure_ascii=True)
        print(line, flush=True)
        if self.handle is not None:
            self.handle.write(line + "\n")
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "JobLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.emit({"stage": "error", "error": repr(exc)})
        self.close()
