"""Run both extractors over the same utterances at increasing noise levels."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from .extract import naive, normalised
from .noise import corrupt
from .slots import VALIDATORS, utterances

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


def load_transcripts(path) -> list[dict]:
    """Rows of {"slot", "value", "transcript"}, optionally with "condition" and "asr":
    what your recogniser actually produced for utterances whose value you know."""
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: not JSON ({e.msg})") from e
        missing = {"slot", "value", "transcript"} - set(row)
        if missing:
            raise ValueError(f"{path}:{n}: missing {sorted(missing)}")
        if row["slot"] not in VALIDATORS:
            raise ValueError(f"{path}:{n}: unknown slot {row['slot']!r}")
        rows.append(row)
    return rows


def evaluate(rows: list[dict], extractors: dict | None = None) -> list[dict]:
    """Correct / wrong / abstain per extractor, for each (asr, condition) group and overall."""
    extractors = extractors or EXTRACTORS
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row.get("asr", "-"), row.get("condition", "-"))
        groups.setdefault(key, []).append(row)
        groups.setdefault(("all", "all"), []).append(row)
    out = []
    for (asr, condition), members in sorted(groups.items()):
        for name, fn in extractors.items():
            total = {"correct": 0, "wrong": 0, "abstain": 0}
            for row in members:
                total[outcome(fn(row["slot"], row["transcript"]), str(row["value"]))] += 1
            out.append(
                {"asr": asr, "condition": condition, "extractor": name, "n": len(members), **total}
            )
    return out


def closest_level(correct_rate: float, curve: dict, extractor: str = "naive") -> float:
    """The synthetic noise level whose accuracy for `extractor` is nearest a measured one:
    a rough way to say which point on the synthetic curve a real recogniser resembles."""
    best, level = 2.0, 0.0
    for row in curve["rows"]:
        if row["extractor"] != extractor:
            continue
        t = row["total"]
        rate = t["correct"] / sum(t.values())
        if abs(rate - correct_rate) < best:
            best, level = abs(rate - correct_rate), row["level"]
    return level
