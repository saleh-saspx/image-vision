import hashlib

from app.utils.hash import compute_sha256


def test_sha256_deterministic():
    data = b"hello world"
    expected = hashlib.sha256(data).hexdigest()
    assert compute_sha256(data) == expected


def test_sha256_different_inputs():
    assert compute_sha256(b"a") != compute_sha256(b"b")
