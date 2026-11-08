"""Real ASR on synthetic speech: what Whisper does to slot values said in Hindi.

Each utterance from `mixslot.slots.utterances` is rewritten as a Devanagari script of how
a caller would say it (order ids letter by letter, digits in Hindi or in English, "double"
for repeats, amounts as Hindi number words), spoken by one of two Piper Hindi voices,
passed through three channel conditions, and transcribed by Whisper. The transcripts are
written to JSON Lines for `mixslot eval`, so the scoring can be re-run without any model.

This is TTS speech, not callers: cleaner, single-speaker, read at an even pace. Real calls
will be worse. It measures what a real recogniser does to numbers and codes, which the
synthetic noise model only guesses at.

Setup (models are GitHub release assets of k2-fsa/sherpa-onnx; sha256 prefixes checked):

    pip install "sherpa-onnx==1.13.8" numpy scipy
    for f in tts-models/vits-piper-hi_IN-pratham-medium \
             tts-models/vits-piper-hi_IN-priyamvada-medium \
             asr-models/sherpa-onnx-whisper-base; do
      curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/$f.tar.bz2
      tar xjf $(basename $f).tar.bz2
    done
    python tools/asr_calibration.py --models . --seed 11 --n 96 -o results/asr-dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
from scipy.signal import butter, resample_poly, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mixslot.slots import utterances  # noqa: E402

HI_DIGITS = "शून्य एक दो तीन चार पाँच छह सात आठ नौ".split()
EN_DIGITS = "ज़ीरो वन टू थ्री फ़ोर फ़ाइव सिक्स सेवन एट नाइन".split()
LETTERS = {"K": "के", "C": "सी"}
# amounts the slot generator uses, in the two ways Hindi speakers say them
AMOUNTS = {
    "299": ["दो सौ निन्यानवे"],
    "499": ["चार सौ निन्यानवे"],
    "799": ["सात सौ निन्यानवे"],
    "999": ["नौ सौ निन्यानवे"],
    "1299": ["एक हज़ार दो सौ निन्यानवे", "बारह सौ निन्यानवे"],
    "1499": ["एक हज़ार चार सौ निन्यानवे", "चौदह सौ निन्यानवे"],
    "1890": ["एक हज़ार आठ सौ नब्बे", "अठारह सौ नब्बे"],
    "2450": ["दो हज़ार चार सौ पचास", "चौबीस सौ पचास"],
    "3200": ["तीन हज़ार दो सौ", "बत्तीस सौ"],
}
TEMPLATES = {
    "order_id": [
        "मेरा ऑर्डर {v} कहाँ तक पहुँचा",
        "ऑर्डर नंबर {v} का स्टेटस बताओ",
        "{v} वाला ऑर्डर कैंसल करना है",
    ],
    "phone": ["मेरा नंबर {v} है, इस पे कॉल करना", "इस नंबर पे व्हाट्सऐप करो {v}"],
    "pin": ["डिलीवरी का पिनकोड {v} है", "पिन कोड {v} पे डिलीवर होता है क्या"],
    "amount": ["मैंने {v} रुपये पे किए थे", "रिफंड {v} रुपये का चाहिए"],
}
VOICES = ["pratham", "priyamvada"]
CONDITIONS = ["clean", "phone", "phone+noise"]


def digits(value: str, r: random.Random, allow_double: bool) -> str:
    words = HI_DIGITS if r.random() < 0.5 else EN_DIGITS
    out, i = [], 0
    while i < len(value):
        if allow_double and i + 1 < len(value) and value[i] == value[i + 1] and r.random() < 0.5:
            out += ["डबल", words[int(value[i])]]
            i += 2
            continue
        out.append(words[int(value[i])])
        i += 1
    return " ".join(out)


def spoken(slot: str, value: str, r: random.Random) -> str:
    if slot == "order_id":
        said = " ".join(LETTERS[c] for c in value[:2]) + " " + digits(value[2:], r, False)
    elif slot == "amount":
        said = r.choice(AMOUNTS[value])
    else:
        said = digits(value, r, allow_double=slot == "phone")
    return r.choice(TEMPLATES[slot]).format(v=said)


def channel(audio: np.ndarray, rate: int, condition: str, r: random.Random) -> np.ndarray:
    x = resample_poly(audio, 16000, rate).astype(np.float32)
    if condition == "clean":
        return x
    # narrowband telephony: 8 kHz, 300-3400 Hz, back to 16 kHz for the recogniser
    nb = resample_poly(x, 1, 2)
    nb = sosfilt(butter(6, [300, 3400], btype="band", fs=8000, output="sos"), nb)
    if condition == "phone+noise":  # white noise at 10 dB SNR
        power = float(np.mean(nb**2)) or 1e-9
        noise = np.random.default_rng(r.randrange(2**32)).normal(0, np.sqrt(power / 10), nb.shape)
        nb = nb + noise
    return resample_poly(nb, 2, 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("."))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--languages", default="hi,en", help="Whisper decoding languages")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    m = args.models
    tts = {
        v: sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(m / f"vits-piper-hi_IN-{v}-medium/hi_IN-{v}-medium.onnx"),
                        tokens=str(m / f"vits-piper-hi_IN-{v}-medium/tokens.txt"),
                        data_dir=str(m / f"vits-piper-hi_IN-{v}-medium/espeak-ng-data"),
                    ),
                    num_threads=2,
                )
            )
        )
        for v in VOICES
    }
    w = m / "sherpa-onnx-whisper-base"
    asr = {
        lang: sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(w / "base-encoder.int8.onnx"),
            decoder=str(w / "base-decoder.int8.onnx"),
            tokens=str(w / "base-tokens.txt"),
            language=lang,
            task="transcribe",
            num_threads=2,
        )
        for lang in args.languages.split(",")
    }
    r = random.Random(f"asr-{args.seed}")
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as f:
        for i, u in enumerate(utterances(args.n, args.seed)):
            text = spoken(u.slot, u.value, r)
            voice = VOICES[i % 2]
            audio = tts[voice].generate(text, sid=0, speed=1.0)
            wave = np.asarray(audio.samples, dtype=np.float32)
            for condition in CONDITIONS:
                x = channel(wave, audio.sample_rate, condition, r)
                for lang, rec in asr.items():
                    s = rec.create_stream()
                    s.accept_waveform(16000, x)
                    rec.decode_stream(s)
                    row = {
                        "id": f"{args.seed}-{i}",
                        "slot": u.slot,
                        "value": u.value,
                        "spoken": text,
                        "voice": voice,
                        "condition": condition,
                        "asr": f"whisper-base/{lang}",
                        "transcript": s.result.text.strip(),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 16 == 15:
                print(f"{i + 1}/{args.n} utterances, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
