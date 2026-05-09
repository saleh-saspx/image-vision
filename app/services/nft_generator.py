import hashlib

from app.models.schemas import Attribute, NFTMetadata

ADJECTIVES = [
    "Neon", "Ethereal", "Crimson", "Phantom", "Crystal", "Shadow",
    "Radiant", "Obsidian", "Celestial", "Midnight", "Golden", "Frozen",
    "Blazing", "Silent", "Emerald", "Violet", "Iron", "Astral",
    "Luminous", "Dusk", "Prismatic", "Onyx", "Sapphire", "Storm",
]

NOUNS = [
    "Ronin", "Specter", "Sentinel", "Oracle", "Wanderer", "Phoenix",
    "Titan", "Voyager", "Cipher", "Nexus", "Druid", "Raven",
    "Monarch", "Eclipse", "Prism", "Golem", "Wraith", "Sage",
    "Vortex", "Beacon", "Herald", "Nomad", "Reaper", "Wisp",
]


def _deterministic_pick(options: list[str], seed: str) -> str:
    idx = int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(options)
    return options[idx]


def _generate_number(image_hash: str) -> str:
    return f"#{int(image_hash[:4], 16) % 1000:03d}"


def generate_name(attributes: dict, image_hash: str) -> str:
    adj_seed = f"{image_hash}:adj:{attributes.get('style', '')}"
    noun_seed = f"{image_hash}:noun:{attributes.get('object_type', '')}"
    adj = _deterministic_pick(ADJECTIVES, adj_seed)
    noun = _deterministic_pick(NOUNS, noun_seed)
    number = _generate_number(image_hash)
    return f"{adj} {noun} {number}"


def generate_description(attributes: dict) -> str:
    subject = attributes.get("object_type", "subject")
    style = attributes.get("style", "unique")
    mood = attributes.get("mood", "evocative")
    lighting = attributes.get("lighting", "ambient")
    environment = attributes.get("environment", "abstract space")

    return (
        f"A {style.lower()}-inspired digital artwork depicting {subject} "
        f"in a {mood.lower()} atmosphere. "
        f"The scene features {lighting.lower()} lighting within {environment.lower()}."
    )


TRAIT_MAP = {
    "style": "Style",
    "dominant_color": "Color",
    "mood": "Mood",
    "lighting": "Lighting",
    "environment": "Background",
    "object_type": "Subject",
}


def generate_attributes(attributes: dict) -> list[Attribute]:
    result = []
    for key, trait_type in TRAIT_MAP.items():
        value = attributes.get(key, "Unknown")
        if value and value != "Unknown":
            result.append(Attribute(trait_type=trait_type, value=value.strip().title()))
    return result


def generate_metadata(attributes: dict, image_hash: str) -> NFTMetadata:
    return NFTMetadata(
        name=generate_name(attributes, image_hash),
        description=generate_description(attributes),
        attributes=generate_attributes(attributes),
    )
