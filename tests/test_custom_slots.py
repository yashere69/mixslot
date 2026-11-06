import pytest

from mixslot.cli import main
from mixslot.extract import extract, normalise


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mera order O R D double four one zero hai", "ORD4410"),
        ("order ess kay you... no wait, ORD 4410", "ORD4410"),
        ("order number ORD44101 hai", None),  # five digits: not this store's format
        ("order kay see 7252", None),  # a different code entirely
    ],
)
def test_custom_pattern(text, expected):
    assert extract(text, r"ORD\d{4}") == expected


def test_spelled_letters_join_only_before_a_number():
    assert normalise("order zed ex nine nine one") == "order ZX991"
    assert normalise("see you at five") == "see you at 5"  # 'see' alone isn't a code
    assert normalise("are you there") == "are you there"


def test_cli_custom_slot(capsys):
    assert main(["extract", "custom", "order O R D 4 4 1 0", "--pattern", r"ORD\d{4}"]) == 0
    assert "value:           ORD4410" in capsys.readouterr().out
    assert main(["extract", "custom", "x"]) == 2
    assert main(["extract", "custom", "x", "--pattern", "("]) == 2


def test_cli_rejects_bad_levels(capsys):
    assert main(["bench", "--levels", "abc"]) == 2
    assert main(["bench", "--levels", "1.5"]) == 2
