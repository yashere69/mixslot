# Changelog

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
