# app/services/text_service.py

import re
from typing import List


def split_text_into_sentences(text: str) -> List[str]:
    """Splits raw text input into clean individual sentences using regex punctuation rules."""
    sentences = re.split(r'(?<=[.!?])\s', text.strip())
    return [s.strip() for s in sentences if s.strip()]