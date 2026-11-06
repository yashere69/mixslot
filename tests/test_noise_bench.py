import random

from mixslot.bench import outcome, run
from mixslot.noise import OPERATORS, corrupt
from mixslot.slots import utterances, valid


def test_zero_noise_is_identity():
    r = random.Random(1)
    for u in utterances(50):
        assert corrupt(u.text, 0.0, r) == u.text


def test_noise_is_deterministic_per_seed():
    texts = [u.text for u in utterances(30)]
    a = [corrupt(t, 0.4, random.Random(5)) for t in texts]
    b = [corrupt(t, 0.4, random.Random(5)) for t in texts]
    assert a == b


def test_every_operator_changes_something_at_full_noise():
    text = "mera order KC7252 aur number 9876543210 hai"
    for name, op in OPERATORS:
        changed = any(op(text, random.Random(s), 1.0) != text for s in range(20))
        assert changed, name


def test_generated_values_are_valid():
    assert all(valid(u.slot, u.value) for u in utterances(200))


def test_outcome():
    assert outcome(None, "x") == "abstain"
    assert outcome("x", "x") == "correct"
    assert outcome("y", "x") == "wrong"


def test_normalising_beats_naive_at_every_noise_level():
    data = run(n=200, seed=3, levels=[0.0, 0.2, 0.4])
    by = {(r["level"], r["extractor"]): r["total"] for r in data["rows"]}
    assert by[(0.0, "naive")]["correct"] == 200
    for level in (0.2, 0.4):
        assert by[(level, "normalised")]["correct"] > by[(level, "naive")]["correct"]
        assert by[(level, "normalised")]["wrong"] <= by[(level, "naive")]["wrong"]
