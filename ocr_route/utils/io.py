"""Helpers for reading and writing common data formats."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def iter_csv_records(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield rows from a CSV file as dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write an iterable of dictionaries to a JSON Lines file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    """Append a single JSON object to a JSON Lines file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Write a dictionary to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_directory(path: str | Path) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def as_list(value: Any) -> List[Any]:
    """Normalize a value to a list."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]
