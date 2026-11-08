"""mixslot eval: scoring transcripts from a real recogniser."""

import json

import pytest

from mixslot import bench
from mixslot.cli import main


def write(tmp_path, rows):
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


def test_eval_counts_correct_wrong_and_abstain(tmp_path, capsys):
    rows = [
        {
            "slot": "order_id",
            "value": "KC1042",
            "transcript": "order KC1042 kahan hai",
            "asr": "x",
            "condition": "clean",
        },
        {
            "slot": "phone",
            "value": "9876543210",
            "transcript": "nau aath saat chhe paanch chaar teen do ek shunya",
            "asr": "x",
            "condition": "clean",
        },
        {
            "slot": "pin",
            "value": "248001",
            "transcript": "pin code 248 0 0 1",
            "asr": "x",
            "condition": "noisy",
        },
        {
            "slot": "order_id",
            "value": "KC1042",
            "transcript": "order KC1043 kahan hai",
            "asr": "x",
            "condition": "noisy",
        },
    ]
    result = bench.evaluate(bench.load_transcripts(write(tmp_path, rows)))
    total = {r["extractor"]: r for r in result if r["asr"] == "all"}
    assert total["naive"]["n"] == 4
    assert (total["normalised"]["correct"], total["normalised"]["wrong"]) == (3, 1)
    assert total["naive"]["correct"] == 1  # only the clean, typed-out order id
    assert main(["eval", str(write(tmp_path, rows))]) == 0
    assert "closest synthetic noise level" in capsys.readouterr().out


@pytest.mark.parametrize(
    "line, message",
    [
        ('{"slot": "order_id", "value": "KC1"}', "missing"),
        ('{"slot": "email", "value": "a", "transcript": "b"}', "unknown slot"),
        ("not json", "not JSON"),
    ],
)
def test_eval_rejects_bad_rows(tmp_path, line, message):
    p = tmp_path / "bad.jsonl"
    p.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        bench.load_transcripts(p)


@pytest.mark.parametrize(
    "slot, transcript, value",
    [
        # forms Whisper produced on spoken Hindi in tools/asr_calibration.py
        ("phone", "मेरा नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है", "9876543210"),
        ("phone", "नंबर नाइन एट सेवन सिक्स फ़ाइव फ़ोर थ्री टू वन ज़ीरो", "9876543210"),
        ("phone", "मेरा नंबर डबल सात एक दो तीन चार पाँच छह सात आठ", "7712345678"),
        ("phone", "میرا نمبر ۹۸۷۶۵۴۳۲۱۰ ہے", "9876543210"),
        ("pin", "پن کوڈ دو چار آٹھ صفر صفر ایک", "248001"),
        ("phone", "call 8-5-8-1-7-5-6-1-2-3", "8581756123"),
        ("order_id", "KC-1042 wala order", "KC1042"),
        ("order_id", "मेरा ऑर्डर के सी एक शून्य चार दो कहाँ", "KC1042"),
    ],
)
def test_scripts_and_separators_from_real_asr_output(slot, transcript, value):
    from mixslot.extract import normalised

    assert normalised(slot, transcript) == value


def test_with_is_not_seven():
    from mixslot.extract import normalise

    assert "7" not in normalise("मैं साथ में आऊँगा")


def test_eval_of_an_empty_file_is_an_error(tmp_path, capsys):
    empty = tmp_path / "t.jsonl"
    empty.write_text("\n", encoding="utf-8")
    assert main(["eval", str(empty)]) == 2
    assert "no transcripts to score" in capsys.readouterr().err
