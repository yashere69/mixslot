"""mixslot command line."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import __version__, bench
from .extract import extract, naive, normalise, normalised


def _table(data: dict) -> str:
    lines = ["noise   extractor    correct   wrong   abstain"]
    for row in data["rows"]:
        t = row["total"]
        n = sum(t.values())
        lines.append(
            f"{row['level']:<7} {row['extractor']:<12} {100 * t['correct'] / n:6.1f}%"
            f" {100 * t['wrong'] / n:6.1f}% {100 * t['abstain'] / n:7.1f}%"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mixslot")
    ap.add_argument("--version", action="version", version=f"mixslot {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="accuracy by noise level for each extractor")
    b.add_argument("--n", type=int, default=400)
    b.add_argument("--seed", type=int, default=11)
    b.add_argument("--levels", help="comma-separated, e.g. 0,0.1,0.3")
    b.add_argument("--json", type=Path)
    b.add_argument("--endpoint", help="also run an OpenAI-compatible model (key: MIXSLOT_API_KEY)")
    b.add_argument("--model")
    e = sub.add_parser("eval", help="score real transcripts whose slot values you know")
    e.add_argument("transcripts", type=Path, help='JSONL: {"slot", "value", "transcript"}')
    e.add_argument("--json", type=Path)
    x = sub.add_parser("extract", help="extract one slot from a transcript")
    x.add_argument("slot", choices=["order_id", "phone", "pin", "amount", "custom"])
    x.add_argument("transcript")
    x.add_argument("--pattern", help="regex for a custom slot, e.g. 'ORD\\d{5}'")
    args = ap.parse_args(argv)

    if args.cmd == "eval":
        try:
            rows = bench.load_transcripts(args.transcripts)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        result = bench.evaluate(rows)
        curve = bench.run(400, 11)
        print("asr                    condition     extractor    n    correct   wrong  abstain")
        for r in result:
            n = r["n"]
            print(
                f"{r['asr']:<22} {r['condition']:<13} {r['extractor']:<11} {n:>4}"
                f" {100 * r['correct'] / n:7.1f}% {100 * r['wrong'] / n:6.1f}%"
                f" {100 * r['abstain'] / n:7.1f}%"
            )
        print("\nclosest synthetic noise level (naive extractor accuracy):")
        for r in result:
            if r["extractor"] == "naive":
                level = bench.closest_level(r["correct"] / r["n"], curve)
                print(f"  {r['asr']:<22} {r['condition']:<13} ~ {level}")
        if args.json:
            args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
        return 0
    if args.cmd == "extract" and args.slot == "custom":
        if not args.pattern:
            print("error: custom slots need --pattern", file=sys.stderr)
            return 2
        try:
            value = extract(args.transcript, args.pattern)
        except re.error as e:
            print(f"error: --pattern: {e}", file=sys.stderr)
            return 2
        print(f"normalised text: {normalise(args.transcript)}")
        print(f"value:           {value}")
        return 0
    if args.cmd == "extract":
        print(f"normalised text: {normalise(args.transcript)}")
        print(f"naive:           {naive(args.slot, args.transcript)}")
        print(f"normalised:      {normalised(args.slot, args.transcript)}")
        return 0

    if args.endpoint and args.model:
        bench.EXTRACTORS["llm"] = bench.llm_extractor(
            args.endpoint, args.model, os.environ.get("MIXSLOT_API_KEY", "")
        )
    try:
        levels = [float(v) for v in args.levels.split(",")] if args.levels else None
    except ValueError:
        print("error: --levels takes numbers like 0,0.1,0.3", file=sys.stderr)
        return 2
    if levels and not all(0 <= lv <= 1 for lv in levels):
        print("error: noise levels are probabilities between 0 and 1", file=sys.stderr)
        return 2
    data = bench.run(args.n, args.seed, levels)
    print(_table(data))
    if args.json:
        args.json.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
