# ====================================================================
#  File: services/memory/token_utils.py
# ====================================================================
"""
Simple token estimation utilities.
This uses a heuristic (average chars-per-token) to avoid heavy deps.
"""

import re
from typing import List

# Heuristic: assume ~4 characters per token for Asian languages may vary.
# Tunable constant.
CHARS_PER_TOKEN = 4.0

# simple whitespace tokenizer fallback
_WORD_RE = re.compile(r"\S+")


def estimate_tokens_for_text(text: str) -> int:
    """Estimate token count for a piece of text.

    This is heuristic and intentionally conservative (slightly overestimates).
    """
    if not text:
        return 0
    # Count non-whitespace characters as a proxy
    chars = len(text)
    est = int(max(1, round(chars / CHARS_PER_TOKEN)))
    return est


def estimate_tokens_for_messages(messages: List[dict]) -> int:
    """Estimate tokens for a list of messages in OpenAI-style format.

    Each message has 'role' and 'content'. We add some overhead per message.
    """
    if not messages:
        return 0
    overhead_per_message = 4  # heuristic tokens for role/content metadata
    total = 0
    for m in messages:
        content = m.get('content', '')
        total += estimate_tokens_for_text(content) + overhead_per_message
    return total
