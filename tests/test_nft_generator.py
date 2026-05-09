from app.services.nft_generator import (
    generate_attributes,
    generate_description,
    generate_metadata,
    generate_name,
)


SAMPLE_ATTRS = {
    "object_type": "warrior",
    "style": "Cyberpunk",
    "dominant_color": "Red",
    "mood": "Dark",
    "lighting": "Neon",
    "environment": "futuristic city",
}

HASH = "aabbccdd" * 8


def test_name_is_deterministic():
    name1 = generate_name(SAMPLE_ATTRS, HASH)
    name2 = generate_name(SAMPLE_ATTRS, HASH)
    assert name1 == name2


def test_name_has_three_parts():
    name = generate_name(SAMPLE_ATTRS, HASH)
    parts = name.split()
    assert len(parts) == 3
    assert parts[-1].startswith("#")


def test_description_mentions_style():
    desc = generate_description(SAMPLE_ATTRS)
    assert "cyberpunk" in desc.lower()


def test_description_not_too_long():
    desc = generate_description(SAMPLE_ATTRS)
    sentences = [s.strip() for s in desc.split(".") if s.strip()]
    assert len(sentences) <= 3


def test_attributes_format():
    attrs = generate_attributes(SAMPLE_ATTRS)
    assert len(attrs) >= 1
    for a in attrs:
        assert a.trait_type
        assert a.value


def test_metadata_returns_full_object():
    meta = generate_metadata(SAMPLE_ATTRS, HASH)
    assert meta.name
    assert meta.description
    assert len(meta.attributes) >= 1


def test_unknown_values_excluded():
    attrs = {"object_type": "Unknown", "style": "Cyberpunk", "dominant_color": "Unknown",
             "mood": "Dark", "lighting": "Unknown", "environment": "Unknown"}
    result = generate_attributes(attrs)
    trait_types = {a.trait_type for a in result}
    assert "Subject" not in trait_types  # object_type=Unknown excluded
    assert "Style" in trait_types
