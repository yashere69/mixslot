"""mixslot command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, bench
from .extract import naive, normalise, normalised


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
    x = sub.add_parser("extract", help="extract one slot from a transcript")
    x.add_argument("slot", choices=["order_id", "phone", "pin", "amount"])
    x.add_argument("transcript")
    args = ap.parse_args(argv)

    if args.cmd == "extract":
        print(f"normalised text: {normalise(args.transcript)}")
        print(f"naive:           {naive(args.slot, args.transcript)}")
        print(f"normalised:      {normalised(args.slot, args.transcript)}")
        return 0

    if args.endpoint and args.model:
        bench.EXTRACTORS["llm"] = bench.llm_extractor(
            args.endpoint, args.model, os.environ.get("MIXSLOT_API_KEY", "")
        )
    levels = [float(v) for v in args.levels.split(",")] if args.levels else None
    data = bench.run(args.n, args.seed, levels)
    print(_table(data))
    if args.json:
        args.json.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
