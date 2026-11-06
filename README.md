# mixslot

When a caller says *"mera order kay see saat do paanch do"*, does the voice agent's
`lookup_order` tool get `KC7252`?

mixslot measures that. It generates Hinglish utterances that carry order ids, phone
numbers, PIN codes and amounts, corrupts them the way speech recognition corrupts spoken
numbers, and checks whether each extractor returns the right value, a **wrong** value, or
**refuses** so the agent can ask again.

**Interactive version:** https://yashere69.github.io/mixslot/

![Slider at noise 0.30: the naive extractor gets 43% right, the normalising one 86.8%, with sample transcripts below](docs/page.png)

## Why this matters

Vendor-published Hinglish speech-to-text comparisons put word error rates for global ASR
models in the high 20s to mid 30s percent on code-switched speech. Those numbers are reported in
words. A voice agent doesn't care about words; it cares whether the 10 digits passed to
`send_whatsapp(phone=...)` are the right 10 digits. One misheard digit in an order id
fails the whole tool call, or worse, finds someone else's order.

Benchmarks stop at the transcript. This one starts there.

## Run it

```bash
pip install git+https://github.com/yashere69/mixslot
mixslot bench --levels 0,0.1,0.3,0.5
mixslot extract phone "mera number nau aath double saat chhe paanch chaar teen do ek hai"
```

No API key needed. To compare a real model as a third extractor:
`MIXSLOT_API_KEY=… mixslot bench --endpoint <openai-compatible-url> --model <name>`.

## Results (400 utterances, seed 11)

| Noise | Naive: correct / wrong / refused | Normalised: correct / wrong / refused |
|---|---|---|
| 0.0 | 400 / 0 / 0 | 400 / 0 / 0 |
| 0.1 | 302 / 16 / 82 | 388 / 0 / 12 |
| 0.3 | 172 / 26 / 202 | 347 / 2 / 51 |
| 0.5 | 83 / 26 / 291 | 321 / 5 / 74 |

Full table in [results/curve.json](results/curve.json); `python tools/build_site.py`
regenerates it along with the page.

The **wrong** column matters more than the correct one. A refusal costs one more turn
("could you repeat the order number?"). A wrong value costs a wrong refund, a message to a
stranger's phone, or a cancelled order that wasn't theirs. The normaliser's validator cuts
wrong values from 26 to 2 at noise 0.3, because a 9-digit "phone number" or a phone
number starting with 5 is refused, not passed on.

## The noise model

Each operator fires with probability proportional to `level` on each eligible token:

| Operator | Example |
|---|---|
| digits spoken as words, Hindi or English, switching mid-number | `9876` → `nau aath seven six` |
| repeats said with "double" | `77` → `double saat` |
| letters spelled out | `KC7252` → `kay see 7252`, `K C 7252` |
| digit groups split by spaces | `9876543210` → `98765 43210` |
| fillers inside numbers | `nau aath haan saat` |
| Devanagari digits in the transcript | `KC७२५२` |
| a digit dropped entirely | `9876543210` → `987654310` |

**The model is not calibrated.** The operators come from how these errors look in
transcripts, not from measured rates for any ASR system, so `level` is a knob, not a WER.
The curves show the shape of the problem and how two extraction strategies compare under
it. To calibrate, you'd need labelled audio from real calls (the HiACC corpus is one
candidate, subject to its licence), which is the obvious next step.

## The normaliser

`mixslot.extract.normalise` rebuilds numbers before any regex sees them:

1. Devanagari digits → ASCII (one-to-one, so nothing shifts)
2. `kay see` / `K C` → `KC`
3. Walks tokens; digit words and digit groups join one run, `double`/`triple` repeat the
   next digit, fillers between digits are dropped and fillers elsewhere are kept
4. Then the slot regex runs, and the value must pass the slot's validator (`KC` + 4 digits;
   10 digits starting 6–9; 6-digit PIN not starting 0) or it's refused

It's around 60 lines and has no model in it. That's the point: in a voice agent, this
belongs between the transcript and the LLM's tool call, where it's cheap, deterministic and
testable.

One detail from writing the tests: Python's `\d` matches Devanagari digits, so a regex alone
happily passes `KC७२५२` to the tool. The naive column counts that as wrong, and a real
database lookup would agree.

## Limits

- **Naive is a floor, not an LLM.** A model reading `nau aath saat…` will do better than a
  regex; use `--endpoint` to measure how much. The normaliser is still useful in front of a
  model, since it hands the model clean digits.
- The utterance templates are short and single-slot. Real calls carry several numbers in one
  turn ("order KC7252, number 98…"), which makes slot confusion possible. That isn't modelled.
- There's no audio anywhere: this is text-only noise.

MIT licensed.
