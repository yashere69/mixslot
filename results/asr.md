# Real ASR on synthetic Hindi speech

Run on 8 November 2026 in a Linux container (2 CPU cores) with `tools/asr_calibration.py`
and scored with `mixslot eval`. The transcripts are committed next to this file, so the
scoring reruns without any model:

    mixslot eval results/asr-dev.jsonl
    mixslot eval results/asr-test.jsonl

**Speech.** Each utterance from `mixslot.slots.utterances` was rewritten as a Devanagari
script of how a caller would say it: order ids letter by letter ("के सी"), digits in Hindi
or in English words ("नौ आठ" / "नाइन एट"), "डबल" for some repeats in phone numbers, and
amounts as Hindi number words ("चौदह सौ निन्यानवे"). Two Piper voices read the scripts:
`hi_IN-pratham-medium` and `hi_IN-priyamvada-medium`, via sherpa-onnx 1.13.8.

**Channel.** Each clip went through three conditions: clean at 16 kHz; `phone` (8 kHz,
300–3400 Hz band-pass); and `phone+noise` (the same plus white noise at 10 dB SNR).

**Recogniser.** Whisper base (multilingual, int8, from sherpa-onnx's release assets),
decoding with `language=hi` and with `language=en`.

**Sets.** `asr-dev` (seed 11) and `asr-test` (seed 23), 96 utterances each, so 576
transcripts per set. The normaliser changes in 0.3.0 were made by reading dev transcripts
only. Test was scored once, afterwards.

## What Whisper base did

It wrote Hindi in Urdu script as often as in Devanagari or Roman letters. It split spoken
digits with hyphens (`8-5-8-17-56`) and merged or dropped digits (`9876543210` came back as
`986654321`). It looped ("pay pay pay pay"), and in noise it produced
`(speaking in foreign language)` or unrelated words. Amounts said in number words came back
as partial digits plus a misspelt word ("200, Nyanbe" for 299).

## Results

Percent of 576 transcripts per set: correct value, **wrong** value passed on, or abstained.

| Set | Extractor | Correct | Wrong | Abstain |
|---|---|---|---|---|
| dev, before 0.3 changes | naive | 5.7 | 7.3 | 87.0 |
| dev, before 0.3 changes | normalised | 8.3 | 8.0 | 83.7 |
| dev, after | normalised | 12.2 | 8.7 | 79.2 |
| **test (held out)** | naive | 6.6 | 12.3 | 81.1 |
| **test (held out)** | normalised | 11.1 | 11.6 | 77.3 |

By condition on test, `language=hi`: clean 18.8 / 15.6 / 65.6 (normalised: correct /
wrong / abstain), phone 17.7 / 9.4 / 72.9, phone+noise 0.0 / 7.3 / 92.7.

## What this says

- **The synthetic curve is optimistic for a small recogniser.** At its harshest level
  (0.5), the synthetic model still leaves the naive extractor more right than it is here.
  `mixslot eval` reports the closest synthetic level for each condition, and for every
  condition it's the top of the scale.
- On held-out transcripts the normaliser raises correct values by about two-thirds over a
  plain regex (6.6% to 11.1%), with the wrong rate no higher (12.3% to 11.6%). Its 0.3 additions all came from dev
  transcripts: Devanagari, English-in-Devanagari and Urdu digit words, Arabic-Indic digits,
  hyphen-split digits, `के सी` letters.
- One in nine values passed on is wrong, mostly amounts and long numbers with a digit
  missing that still pass the validator. A voice agent should read numbers back before
  acting on them. No extractor fixes a recogniser that didn't hear the digit.

## Limits

TTS speech is cleaner, slower and more regular than callers on a phone line. Whisper base
is far weaker than production recognisers. Run `mixslot eval` on transcripts from yours,
since the result depends mostly on the recogniser. One run per seed; no confidence
intervals.
