import pytest

from mixslot.extract import naive, normalise, normalised


@pytest.mark.parametrize(
    "text, expected",
    [
        ("order kay see saat do paanch do", "order KC7252"),
        ("order K C 7252", "order KC7252"),
        ("call nau aath double saat chhe paanch chaar teen do ek", "call 9877654321"),
        ("call nine eight seven six five haan four three two one zero", "call 9876543210"),
        ("call 98765 43210", "call 9876543210"),
        ("pin २४८००१", "pin 248001"),
        ("do kurte chahiye", "2 kurte chahiye"),
    ],
)
def test_normalise(text, expected):
    assert normalise(text) == expected


def test_trailing_filler_is_kept_when_no_digit_follows():
    assert normalise("number nau haan") == "number 9 haan"


@pytest.mark.parametrize(
    "slot, text, gold",
    [
        ("order_id", "mera order kay see saat do paanch do kahan hai", "KC7252"),
        (
            "phone",
            "mera number nau aath double saat chhe paanch chaar teen do ek hai",
            "9877654321",
        ),
        ("pin", "pincode ２४८ ००१".replace("２", "2"), "248001"),
        ("amount", "maine paanch sau... 1499 rupaye diye", "1499"),
    ],
)
def test_normalised_recovers(slot, text, gold):
    assert normalised(slot, text) == gold


def test_validator_refuses_instead_of_guessing():
    # a dropped digit leaves 9 digits; naive finds nothing, normalised refuses
    assert naive("phone", "call 987654321") is None
    assert normalised("phone", "call 987654321") is None
    # a phone number cannot start with 5
    assert normalised("phone", "call 5876543210") is None


def test_naive_passes_devanagari_digits_through_unchanged():
    # Python's \\d matches Devanagari digits, so a regex alone hands the tool "KC७२५२"
    assert naive("order_id", "order KC७२५२") == "KC७२५२"
    assert normalised("order_id", "order KC७२५२") == "KC7252"
