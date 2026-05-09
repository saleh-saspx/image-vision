from app.core.cache import LRUCache
from app.models.schemas import Attribute, NFTMetadata, NFTResponse


def _dummy_response(hash_val: str = "abc") -> NFTResponse:
    return NFTResponse(
        image_hash=hash_val,
        metadata=NFTMetadata(
            name="Test",
            description="Test desc",
            attributes=[Attribute(trait_type="Style", value="Test")],
        ),
    )


def test_cache_miss_returns_none():
    c = LRUCache(max_size=4)
    assert c.get("missing") is None


def test_cache_hit():
    c = LRUCache(max_size=4)
    resp = _dummy_response("x")
    c.set("x", resp)
    assert c.get("x") == resp


def test_cache_eviction():
    c = LRUCache(max_size=2)
    c.set("a", _dummy_response("a"))
    c.set("b", _dummy_response("b"))
    c.set("c", _dummy_response("c"))  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") is not None
    assert c.get("c") is not None


def test_cache_lru_order():
    c = LRUCache(max_size=2)
    c.set("a", _dummy_response("a"))
    c.set("b", _dummy_response("b"))
    c.get("a")  # touch "a", making "b" the LRU
    c.set("c", _dummy_response("c"))  # evicts "b"
    assert c.get("a") is not None
    assert c.get("b") is None
