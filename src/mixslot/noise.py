"""A transcript noise model for code-switched (Hindi-English) speech recognition.

Each operator imitates something an ASR system does to numbers and codes said aloud in
Hinglish. `level` is the probability that each operator fires on each eligible token, so
level 0 is a perfect transcript.

This model is not calibrated against a real ASR system; the operator list comes from how
these errors look, not from measured rates. The curves it produces show the *shape* of the
problem and compare extraction strategies. They aren't a WER-to-accuracy mapping for any
vendor.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
HI = ["shunya", "ek", "do", "teen", "chaar", "paanch", "chhe", "saat", "aath", "nau"]
DEV_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")
LETTER_NAMES = {"K": "kay", "C": "see"}
FILLERS = ["haan", "matlab", "uh", "acha", "ki"]

_DIGIT_RUN = re.compile(r"\d+")


def _spell_digits(run: str, r: random.Random, level: float) -> str:
    """Speak a digit run: words in Hindi or English, 'double' for repeats, sometimes mixed."""
    out: list[str] = []
    i = 0
    lang = HI if r.random() < 0.5 else EN
    while i < len(run):
        d = run[i]
        if i + 1 < len(run) and run[i + 1] == d and r.random() < level:
            out += ["double", (lang[int(d)] if r.random() < 0.7 else d)]
            i += 2
            continue
        if r.random() < level / 2:
            lang = HI if lang is EN else EN  # code-switch mid-number
        out.append(lang[int(d)])
        i += 1
    return " ".join(out)


def op_words(text: str, r: random.Random, level: float) -> str:
    return _DIGIT_RUN.sub(
        lambda m: _spell_digits(m.group(), r, level) if r.random() < level else m.group(), text
    )


def op_group(text: str, r: random.Random, level: float) -> str:
    """ASR inserts spaces inside long numbers: 9876543210 -> 98765 43210."""

    def split(m: re.Match[str]) -> str:
        s = m.group()
        if len(s) < 6 or r.random() >= level:
            return s
        cut = r.choice([2, 3, 5]) if len(s) > 6 else 3
        return s[:cut] + " " + s[cut:]

    return _DIGIT_RUN.sub(split, text)


def op_letters(text: str, r: random.Random, level: float) -> str:
    """'KC1042' heard as 'K C 1042' or 'kay see 1042'."""

    def spell(m: re.Match[str]) -> str:
        if r.random() >= level:
            return m.group()
        if r.random() < 0.5:
            return "K C " + m.group(1)
        return f"{LETTER_NAMES['K']} {LETTER_NAMES['C']} {m.group(1)}"

    return re.sub(r"\bKC(\d+)", spell, text)


def op_fillers(text: str, r: random.Random, level: float) -> str:
    words = text.split(" ")
    out = []
    for w in words:
        out.append(w)
        if r.random() < level / 3:
            out.append(r.choice(FILLERS))
    return " ".join(out)


def op_devanagari(text: str, r: random.Random, level: float) -> str:
    return text.translate(DEV_DIGITS) if r.random() < level / 2 else text


def op_drop(text: str, r: random.Random, level: float) -> str:
    """A digit that never made it into the transcript. Not recoverable from text alone."""

    def drop(m: re.Match[str]) -> str:
        s = m.group()
        if len(s) < 4 or r.random() >= level / 4:
            return s
        i = r.randrange(len(s))
        return s[:i] + s[i + 1 :]

    return _DIGIT_RUN.sub(drop, text)


OPERATORS: list[tuple[str, Callable[[str, random.Random, float], str]]] = [
    ("drop", op_drop),
    ("letters", op_letters),
    ("group", op_group),
    ("words", op_words),
    ("fillers", op_fillers),
    ("devanagari", op_devanagari),
]


def corrupt(text: str, level: float, r: random.Random) -> str:
    for _, op in OPERATORS:
        text = op(text, r, level)
    return text
