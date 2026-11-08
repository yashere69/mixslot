"""Two ways to pull a slot value out of a transcript.

`naive` is the regex you'd write against clean text, the same thing an LLM-generated tool
argument amounts to when the model copies whatever digits it sees. `normalised` first
rebuilds numbers the way they were said (words, "double", spaced groups, spelled letters,
Devanagari digits, fillers in between), then refuses any value that fails the slot's
validator. Refusing turns a wrong order id into "could you repeat that?", which is the
whole point.
"""

from __future__ import annotations

import re

from .noise import EN, FILLERS, HI
from .slots import VALIDATORS

WORD_TO_DIGIT = {w: str(i) for i, w in enumerate(EN)} | {w: str(i) for i, w in enumerate(HI)}
WORD_TO_DIGIT |= {"oh": "0", "char": "4", "panch": "5", "chah": "6", "sat": "7", "ath": "8"}
# Recognisers write Hindi speech in Devanagari, in Urdu script, or as English words spelled
# in Devanagari ("वन टू"). Digit words in all three, checked against Whisper's output on
# spoken Hindi (tools/asr_calibration.py). "साथ"/"ساتھ" (with) is left out on purpose: it is
# far more often the word than a misheard seven.
WORD_TO_DIGIT |= {w: str(i) for i, w in enumerate("शून्य एक दो तीन चार पाँच छह सात आठ नौ".split())}
WORD_TO_DIGIT |= {
    "पांच": "5",
    "छः": "6",
    "छे": "6",
    "ज़ीरो": "0",
    "जीरो": "0",
    "वन": "1",
    "टू": "2",
    "थ्री": "3",
    "फ़ोर": "4",
    "फोर": "4",
    "फ़ाइव": "5",
    "फाइव": "5",
    "सिक्स": "6",
    "सेवन": "7",
    "एट": "8",
    "नाइन": "9",
}
WORD_TO_DIGIT |= {w: str(i) for i, w in enumerate("صفر ایک دو تین چار پانچ چھ سات آٹھ نو".split())}
REPEAT = {"double": 2, "triple": 3, "डबल": 2, "ट्रिपल": 3, "ڈبل": 2}
# Devanagari, Arabic-Indic and Extended Arabic-Indic (Urdu) digits
DEV = str.maketrans("०१२३४५६७८९٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "012345678901234567890123456789")
LETTER_WORDS = {"के": "K", "सी": "C", "کے": "K", "سی": "C"}

PATTERNS = {
    "order_id": re.compile(r"KC\s?(\d{4})\b"),
    "phone": re.compile(r"(?<!\d)([6-9]\d{9})(?!\d)"),
    "pin": re.compile(r"(?<!\d)(\d{6})(?!\d)"),
    "amount": re.compile(r"(?<!\d)(\d{2,6})(?!\d)"),
}


def naive(slot: str, text: str) -> str | None:
    m = PATTERNS[slot].search(text)
    if not m:
        return None
    return ("KC" + m.group(1)) if slot == "order_id" else m.group(1)


# Letter names ASR writes when a caller spells a code. "are", "you", "why" and "oh" are left
# out on purpose: they're ordinary words far more often than they're R, U, Y or O.
LETTER_NAMES = {
    "ay": "A",
    "bee": "B",
    "see": "C",
    "dee": "D",
    "ee": "E",
    "ef": "F",
    "jee": "G",
    "aitch": "H",
    "jay": "J",
    "kay": "K",
    "el": "L",
    "em": "M",
    "en": "N",
    "pee": "P",
    "queue": "Q",
    "cue": "Q",
    "ess": "S",
    "tee": "T",
    "vee": "V",
    "ex": "X",
    "zed": "Z",
    "zee": "Z",
}


def _letter(tok: str) -> str | None:
    if len(tok) == 1 and tok.isalpha() and tok.isupper():
        return tok
    return LETTER_NAMES.get(tok.lower()) or LETTER_WORDS.get(tok)


# Digits read out one at a time often come back hyphenated ("8-5-8-17-56") and a code with
# a hyphen after its letters ("KC-0387"). Both are split into the spaced form the rest of
# the normaliser already joins.
_HYPHEN_DIGITS = re.compile(r"(?<=\d)-(?=\d)|(?<=\b[A-Z]{2})-(?=\d)")


def _spelled_codes(tokens: list[str]) -> list[str]:
    """Join a spelled run of two or more letters that is followed by a number:
    'kay see 7252' -> 'KC 7252', 'O R D 4410' stays letters only when digits follow."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j = i
        letters = []
        while j < len(tokens) and (ch := _letter(tokens[j].strip(",."))):
            letters.append(ch)
            j += 1
        nxt = tokens[j].lower().strip(",.") if j < len(tokens) else ""
        if len(letters) >= 2 and (nxt.isdigit() or nxt in WORD_TO_DIGIT or nxt in REPEAT):
            out.append("\x01" + "".join(letters))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def normalise(text: str) -> str:
    """Rebuild codes and numbers the way they were said. Deterministic, no model."""
    tokens = _spelled_codes(_HYPHEN_DIGITS.sub(" ", text.translate(DEV)).split())
    out: list[str] = []
    run = ""
    repeat = 1
    pending_fillers: list[str] = []

    def flush() -> None:
        nonlocal run
        if run:
            out.append(run)
            run = ""

    for tok in tokens:
        low = tok.lower().strip(",.")
        if low in REPEAT:
            repeat = REPEAT[low]
            continue
        digit = WORD_TO_DIGIT.get(low)
        if digit is None and low.isdigit():
            digit = low
        if digit is not None:
            pending_fillers = []  # fillers between digits are dropped
            run += digit * repeat if len(digit) == 1 else digit
            repeat = 1
            continue
        if run and low in FILLERS:
            pending_fillers.append(tok)
            continue
        flush()
        out.extend(pending_fillers)
        pending_fillers = []
        repeat = 1
        out.append(tok)
    flush()
    out.extend(pending_fillers)
    joined = " ".join(out)
    joined = re.sub(r"\x01([A-Z]+) (\d)", r"\1\2", joined)  # spelled letters + their digits
    return joined.replace("\x01", "")


def extract(text: str, pattern: str | re.Pattern[str]) -> str | None:
    r"""Your own slot: normalise, then return the first match of `pattern`, or None.

    `pattern` is matched against the normalised text with digit boundaries on both sides,
    so r"ORD\d{5}" won't match inside a longer number. None means "ask the caller again".
    """
    rx = re.compile(pattern) if isinstance(pattern, str) else pattern
    norm = normalise(text)
    # also try with a typed space closed up ('ORD 4410'), without rewriting the text itself
    for candidate in (norm, re.sub(r"\b([A-Z]{2,4}) (\d)", r"\1\2", norm)):
        for m in rx.finditer(candidate):
            before = candidate[m.start() - 1] if m.start() else " "
            after = candidate[m.end()] if m.end() < len(candidate) else " "
            if not before.isalnum() and not after.isdigit():
                return m.group()
    return None


def normalised(slot: str, text: str) -> str | None:
    value = naive(slot, normalise(text))
    if value is None or not VALIDATORS[slot].match(value):
        return None
    return value
