# ====================================================================
#  File: services/memory/token_utils.py
# ====================================================================
import re
from typing import List

CHARS_PER_TOKEN = 4.0

_WORD_RE = re.compile(r"\S+")


def estimate_tokens_for_text(text: str) -> int:
    if not text:
        return 0
    chars = len(text)
    est = int(max(1, round(chars / CHARS_PER_TOKEN)))
    return est


def estimate_tokens_for_messages(messages: List[dict]) -> int:
    if not messages:
        return 0
    overhead_per_message = 4
    total = 0
    for m in messages:
        content = m.get('content', '')
        total += estimate_tokens_for_text(content) + overhead_per_message
    return total
