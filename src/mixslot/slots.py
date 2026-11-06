"""Slot types a voice agent for a small shop has to get exactly right, and utterances
that carry them. Each slot has a validator; a value that fails it is never acted on."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Utterance:
    text: str
    slot: str
    value: str


VALIDATORS = {
    "order_id": re.compile(r"^KC\d{4}$"),
    "phone": re.compile(r"^[6-9]\d{9}$"),
    "pin": re.compile(r"^[1-9]\d{5}$"),
    "amount": re.compile(r"^[1-9]\d{1,5}$"),
}

TEMPLATES = {
    "order_id": [
        "mera order {v} kahan tak pahuncha",
        "order number {v} ka status batao",
        "{v} wala order cancel karna hai",
        "where is my order {v}",
    ],
    "phone": [
        "mera number {v} hai, ispe call karna",
        "is number pe whatsapp karo {v}",
        "you can reach me at {v}",
    ],
    "pin": [
        "delivery ka pincode {v} hai",
        "pin code {v} pe deliver hota hai kya",
    ],
    "amount": [
        "maine {v} rupaye pay kiye the",
        "refund {v} rupees ka chahiye",
    ],
}


def value_for(slot: str, r: random.Random) -> str:
    if slot == "order_id":
        return f"KC{r.randint(1000, 9999)}"
    if slot == "phone":
        return r.choice("6789") + "".join(str(r.randint(0, 9)) for _ in range(9))
    if slot == "pin":
        return f"{r.randint(1, 8)}{r.randint(0, 99999):05d}"
    return str(r.choice([299, 499, 799, 999, 1299, 1499, 1890, 2450, 3200]))


def utterances(n: int, seed: int = 11) -> list[Utterance]:
    r = random.Random(seed)
    out = []
    for _ in range(n):
        slot = r.choice(sorted(TEMPLATES))
        value = value_for(slot, r)
        out.append(Utterance(r.choice(TEMPLATES[slot]).format(v=value), slot, value))
    return out


def valid(slot: str, value: str | None) -> bool:
    return bool(value) and bool(VALIDATORS[slot].match(value or ""))
