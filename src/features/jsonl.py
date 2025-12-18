"""JSONL helpers for gating and calibration datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into memory."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_get(d: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Retrieve a value from nested dicts using dotted access."""
    cur: Any = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def get_label(row: Dict[str, Any]) -> Optional[int]:
    """Return label_call_ocr_good as {0,1} if present."""
    value = row.get("label_call_ocr_good")
    if value is None:
        return None
    return 1 if bool(value) else 0

