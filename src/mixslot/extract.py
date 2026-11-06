"""Two ways to pull a slot value out of a transcript.

`naive` is the regex you'd write against clean text, the same thing an LLM-generated tool
argument amounts to when the model copies whatever digits it sees. `normalised` first
rebuilds numbers the way they were said (words, "double", spaced groups, spelled letters,
Devanagari digits, fillers in between), then refuses any value that fails the slot's
validator. Refusing turns a wrong order id into "could you repeat that?", which is the
whole point.
"""

from __future__ import annotations

import re

from .noise import EN, FILLERS, HI
from .slots import VALIDATORS

WORD_TO_DIGIT = {w: str(i) for i, w in enumerate(EN)} | {w: str(i) for i, w in enumerate(HI)}
WORD_TO_DIGIT |= {"oh": "0", "char": "4", "panch": "5", "chah": "6", "sat": "7", "ath": "8"}
REPEAT = {"double": 2, "triple": 3}
DEV = str.maketrans("०१२३४५६७८९", "0123456789")

PATTERNS = {
    "order_id": re.compile(r"KC\s?(\d{4})\b"),
    "phone": re.compile(r"(?<!\d)([6-9]\d{9})(?!\d)"),
    "pin": re.compile(r"(?<!\d)(\d{6})(?!\d)"),
    "amount": re.compile(r"(?<!\d)(\d{2,6})(?!\d)"),
}


def naive(slot: str, text: str) -> str | None:
    m = PATTERNS[slot].search(text)
    if not m:
        return None
    return ("KC" + m.group(1)) if slot == "order_id" else m.group(1)


def normalise(text: str) -> str:
    text = text.translate(DEV)
    text = re.sub(r"\b(kay|k)\s+(see|c)\b", "KC", text, flags=re.I)
    tokens = text.split()
    out: list[str] = []
    run = ""
    repeat = 1
    pending_fillers: list[str] = []

    def flush() -> None:
        nonlocal run
        if run:
            out.append(run)
            run = ""

    for tok in tokens:
        low = tok.lower().strip(",.")
        if low in REPEAT:
            repeat = REPEAT[low]
            continue
        digit = WORD_TO_DIGIT.get(low)
        if digit is None and low.isdigit():
            digit = low
        if digit is not None:
            pending_fillers = []  # fillers between digits are dropped
            run += digit * repeat if len(digit) == 1 else digit
            repeat = 1
            continue
        if run and low in FILLERS:
            pending_fillers.append(tok)
            continue
        flush()
        out.extend(pending_fillers)
        pending_fillers = []
        repeat = 1
        out.append(tok)
    flush()
    out.extend(pending_fillers)
    return re.sub(r"\bKC\s+(\d)", r"KC\1", " ".join(out))


def normalised(slot: str, text: str) -> str | None:
    value = naive(slot, normalise(text))
    if value is None or not VALIDATORS[slot].match(value):
        return None
    return value
