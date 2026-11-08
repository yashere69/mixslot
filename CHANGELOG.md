# Changelog

## 0.3.1

- Output is UTF-8 even when redirected on Windows, where it used to follow the code page
  and fail on Devanagari text.
- CI runs the tests on Windows and macOS as well as Linux.
- `eval` on a file with no transcripts is an error; it used to print an empty table and
  exit 0.

## 0.3.0

- `mixslot eval transcripts.jsonl`: score transcripts from a real recogniser (correct,
  wrong, abstain per condition) and report the closest synthetic noise level.
- `tools/asr_calibration.py`: Hindi TTS (two Piper voices), three channel conditions,
  Whisper base. Dev and held-out transcripts and `results/asr.md` are committed.
- The normaliser reads digit words in Devanagari, Urdu script and English-in-Devanagari,
  Arabic-Indic digits, `डबल`, `के सी`, and hyphen-split digits (`8-5-8…`, `KC-1042`). Each
  form was seen in Whisper's output. On the held-out set it gets 11.1% of values right
  against the regex's 6.6%.

## 0.2.0

- `extract(text, pattern)` and `mixslot extract custom --pattern` for your own code formats
- Spelled letter runs ("O R D", "zed ex") joined generally, not just "kay see"
- `--levels` validated (numbers between 0 and 1)

## 0.1.0

- Slot generator (order id, phone, PIN, amount) with Hinglish templates
- Six-operator transcript noise model
- Naive and normalising extractors; validator refuses instead of guessing
- `mixslot bench`, `mixslot extract`, optional OpenAI-compatible extractor
- Interactive results page, deployed to GitHub Pages from CI
