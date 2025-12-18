"""Text processing helpers for post-processing model outputs."""
from __future__ import annotations

import re
from typing import Optional


def extract_assistant_answer(text: Optional[str]) -> str:
    """Strip chat-template headers and return the assistant response body."""
    if not text:
        return ""
    # If model followed the ANSWER: [ANSWER] format, extract it.
    m_answer = re.search(r'ANSWER:\s*\[?\s*([^\]\n]+?)\s*\]?\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
    if m_answer:
        ans = m_answer.group(1).strip()
        if ans:
            return ans
    # If the model followed the format hint, extract the final answer only.
    m = re.search(r'final answer is\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        # If the model left the placeholder unchanged, fall back to the raw text parsing below.
        if ans not in {"<answer>", "<final_answer>", "answer"}:
            return ans
    marker = "\nassistant\n"
    if marker in text:
        return text.rsplit(marker, maxsplit=1)[-1].strip()
    # Fallbacks for slightly different templates.
    if text.startswith("assistant\n"):
        return text[len("assistant\n") :].strip()
    if "\nassistant:" in text:
        return text.split("\nassistant:", maxsplit=1)[-1].strip()
    return text.strip()
