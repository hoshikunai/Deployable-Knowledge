from __future__ import annotations

import re
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".cache" / "relex-eval"
SET_PATH = OUT / "2wiki-set.json"
PREDICTIONS_PATH = OUT / "2wiki-ts.jsonl"
ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCTUATION = set(string.punctuation)


def normalize(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(character for character in lowered if character not in PUNCTUATION)
    return " ".join(ARTICLES.sub(" ", without_punctuation).split())


def token_f1(left: str, right: str) -> float:
    left_tokens, right_tokens = normalize(left).split(), normalize(right).split()
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    shared = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if shared == 0:
        return 0.0
    precision, recall = shared / len(right_tokens), shared / len(left_tokens)
    return 2 * precision * recall / (precision + recall)


def same_entity(gold: str, predicted: str, lenient: bool) -> bool:
    expected, found = normalize(gold), normalize(predicted)
    if not expected or not found:
        return False
    if expected == found:
        return True
    if not lenient:
        return False
    return expected in found or found in expected or token_f1(gold, predicted) >= 0.5


def mentioned(entity: str, text: str) -> bool:
    expected = normalize(entity)
    return bool(expected) and f" {expected} " in f" {normalize(text)} "
