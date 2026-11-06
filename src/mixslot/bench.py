"""Run both extractors over the same utterances at increasing noise levels."""

from __future__ import annotations

import random
from collections.abc import Callable

from .extract import naive, normalised
from .noise import corrupt
from .slots import utterances

EXTRACTORS: dict[str, Callable[[str, str], str | None]] = {
    "naive": naive,
    "normalised": normalised,
}
LEVELS = [round(x * 0.05, 2) for x in range(0, 11)]


def outcome(got: str | None, gold: str) -> str:
    if got is None:
        return "abstain"
    return "correct" if got == gold else "wrong"


def run(n: int = 400, seed: int = 11, levels: list[float] | None = None) -> dict:
    utts = utterances(n, seed)
    rows = []
    for level in levels or LEVELS:
        r = random.Random(f"{seed}-{level}")
        noisy = [corrupt(u.text, level, r) for u in utts]
        for name, fn in EXTRACTORS.items():
            per_slot: dict[str, dict[str, int]] = {}
            for u, t in zip(utts, noisy, strict=True):
                c = per_slot.setdefault(u.slot, {"correct": 0, "wrong": 0, "abstain": 0})
                c[outcome(fn(u.slot, t), u.value)] += 1
            total = {
                k: sum(c[k] for c in per_slot.values()) for k in ("correct", "wrong", "abstain")
            }
            rows.append({"level": level, "extractor": name, "total": total, "slots": per_slot})
    return {"n": n, "seed": seed, "rows": rows}


def examples(level: float = 0.3, n: int = 6, seed: int = 11) -> list[dict]:
    r = random.Random(f"{seed}-examples")
    out = []
    for u in utterances(n, seed + 1):
        noisy = corrupt(u.text, level, r)
        out.append(
            {
                "slot": u.slot,
                "gold": u.value,
                "transcript": noisy,
                "naive": naive(u.slot, noisy),
                "normalised": normalised(u.slot, noisy),
            }
        )
    return out


def llm_extractor(endpoint: str, model: str, key: str) -> Callable[[str, str], str | None]:
    """An extractor backed by any OpenAI-compatible chat endpoint, for comparing a real model
    against the two above. Not used in tests or CI."""
    import json
    import urllib.request

    def extract(slot: str, text: str) -> str | None:
        prompt = (
            f"Extract the {slot.replace('_', ' ')} from this voice-call transcript. Reply with "
            "only the value, digits and capital letters, or NONE if it isn't there.\n\n" + text
        )
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            answer = json.load(r)["choices"][0]["message"]["content"].strip()
        return None if answer.upper() == "NONE" else answer.replace(" ", "")

    return extract
