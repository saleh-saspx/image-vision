"""Turns raw model output + pixel facts into reviewable NFT metadata.

Design rule: the model supplies semantics, the pixels supply measurements, and
this module supplies the writing. Nothing here invents a fact — every sentence,
tag and trait is assembled from a field that was actually observed, and each one
carries the confidence of its weakest input.
"""
from __future__ import annotations

import re

from app.core import vocab
from app.models.schemas import Attribute, NFTMetadata
from app.services.imagestats import ImageStats

# ---------------------------------------------------------------------------
# Confidence model
# ---------------------------------------------------------------------------
# Ceiling of 0.9 is reserved for facts read straight off the pixels. A generated
# token never gets there: even a confident-sounding 2B model is guessing about
# style and mood, and over-scoring it would defeat the point of review.
CEILING = 0.9

_TIER_CONFIDENCE = {
    vocab.MATCH_EXACT: 0.78,       # model said a term we recognise
    vocab.MATCH_FUZZY: 0.66,       # we mapped it onto one
    vocab.MATCH_PASSTHROUGH: 0.52,  # we are echoing free text
    vocab.MATCH_NONE: 0.0,
}

# Measured, not guessed.
CONF_MEASURED = 0.88
CONF_DERIVED_STRONG = 0.7   # rule-derived from several agreeing fields
CONF_DERIVED_WEAK = 0.45    # rule-derived from one weak signal

_SUBJECT_STOP = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def _clean_subject(raw: str) -> str:
    text = vocab.clean_text(raw)
    if not text:
        return ""
    text = _SUBJECT_STOP.sub("", text)
    return vocab._titleize(text)


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------

_SCALAR_FIELDS = {
    "style": "style",
    "art_medium": "art_medium",
    "scene": "scene",
    "environment": "environment",
    "lighting": "lighting",
    "mood": "mood",
    "perspective": "perspective",
    "texture": "texture",
    "pattern": "pattern",
}


def normalize_fields(raw: dict) -> tuple[dict, dict[str, float]]:
    """Map raw model output onto the controlled vocabulary, scoring each field."""
    fields: dict[str, object] = {}
    confidence: dict[str, float] = {}

    for field, vocab_name in _SCALAR_FIELDS.items():
        value, tier = vocab.normalize(vocab_name, raw.get(field))
        if value:
            fields[field] = value
            confidence[field] = _TIER_CONFIDENCE[tier]

    subject = _clean_subject(raw.get("primary_subject", ""))
    if subject:
        fields["primary_subject"] = subject
        # Subjects are open-vocabulary, so there is nothing to match against;
        # a plausible-looking noun phrase is worth a middling score at best.
        confidence["primary_subject"] = 0.6

    secondary = _dedupe(
        [_clean_subject(s) for s in raw.get("secondary_subjects", []) or []],
        exclude={subject},
    )
    if secondary:
        fields["secondary_subjects"] = secondary[:6]
        confidence["secondary_subjects"] = 0.55

    objects = _dedupe([_clean_subject(o) for o in raw.get("objects", []) or []])
    if objects:
        fields["objects"] = objects[:15]
        confidence["objects"] = 0.58

    materials, material_tiers = [], []
    for item in raw.get("materials", []) or []:
        value, tier = vocab.normalize("material", item)
        if value and value not in materials:
            materials.append(value)
            material_tiers.append(_TIER_CONFIDENCE[tier])
    if materials:
        fields["materials"] = materials[:5]
        # Materials are an inference from surface appearance; never a strong one.
        confidence["materials"] = round(min(sum(material_tiers) / len(material_tiers), 0.7), 2)

    return fields, confidence


def _dedupe(items: list[str], exclude: set[str] | None = None) -> list[str]:
    exclude = {e.lower() for e in (exclude or set()) if e}
    seen, out = set(), []
    for item in items:
        key = item.lower()
        if item and key not in seen and key not in exclude:
            seen.add(key)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Pixel-measured fields
# ---------------------------------------------------------------------------

def apply_image_stats(fields: dict, confidence: dict[str, float], stats: ImageStats) -> None:
    if stats.colors:
        fields["dominant_colors"] = stats.colors
        # Confidence tracks how much of the frame the top colour actually owns:
        # a two-tone graphic is certain, a muddy photo genuinely is not.
        top_share = stats.color_weights[0] if stats.color_weights else 0.0
        confidence["dominant_colors"] = round(min(CEILING, 0.7 + top_share * 0.25), 2)

    fields["visual_complexity"] = stats.complexity
    confidence["visual_complexity"] = CONF_MEASURED


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

# Ordered rules: first hit wins.
#
# Subject-driven categories deliberately outrank medium-driven ones: a 3D render
# of a sofa belongs in Furniture, because "3D Render" is already carried by the
# art_medium facet and repeating it as the category costs a filtering axis.
_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Portrait", ("portrait", "face", "man", "woman", "person", "girl", "boy", "head")),
    ("Animal", ("cat", "dog", "bird", "horse", "lion", "tiger", "fish", "animal", "creature", "ape", "monkey")),
    ("Vehicle", ("car", "vehicle", "motorcycle", "ship", "plane", "aircraft", "truck", "spaceship")),
    ("Furniture", ("sofa", "chair", "table", "couch", "desk", "cabinet", "bed", "shelf", "lamp", "vase", "furniture")),
    ("Interior", ("living room", "bedroom", "kitchen", "dining room", "bathroom", "office", "interior")),
    ("Architecture", ("building", "house", "tower", "bridge", "temple", "castle", "architecture", "cathedral")),
    ("Landscape", ("mountain", "forest", "beach", "desert", "ocean", "sky", "landscape", "valley", "lake")),
    ("Fashion", ("dress", "shoe", "jacket", "hat", "clothing", "outfit", "bag", "watch")),
    ("Food", ("food", "fruit", "drink", "coffee", "cake", "meal", "bottle")),
    ("Fantasy", ("dragon", "wizard", "elf", "knight", "fantasy", "magical", "mythical")),
    ("Nature", ("plant", "flower", "tree", "leaf", "nature")),
    ("Abstract", ("abstract", "geometric", "shapes")),
    ("Gaming", ("game", "gaming", "avatar", "character")),
    # Medium-driven fallbacks, only reached when the subject says nothing.
    ("Pixel Art", ("pixel art",)),
    ("Anime", ("anime", "manga")),
    ("3D", ("3d render", "low poly")),
    ("Photography", ("photography",)),
]


# Whole-word matching is mandatory here: plain substring search puts a "street"
# scene into Nature because it contains "tree".
_CATEGORY_PATTERNS = [
    (category, re.compile(r"\b(?:%s)\b" % "|".join(re.escape(k) for k in keywords)))
    for category, keywords in _CATEGORY_RULES
]


def infer_category(fields: dict) -> tuple[str | None, float]:
    haystack = " ".join(
        str(fields.get(key, "")).lower()
        for key in ("primary_subject", "scene", "style", "art_medium", "environment")
    )
    haystack += " " + " ".join(o.lower() for o in fields.get("objects", []) or [])

    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(haystack):
            return category, CONF_DERIVED_STRONG

    # Fall back on medium: a photo is Photography, anything else is Art.
    medium = fields.get("art_medium")
    if medium == "Photography":
        return "Photography", CONF_DERIVED_WEAK
    if medium:
        return "Art", CONF_DERIVED_WEAK
    return None, 0.0


def infer_composition(fields: dict, stats: ImageStats) -> tuple[str, float]:
    """Composition is inferable from geometry and object count without asking."""
    object_count = len(fields.get("objects", []) or [])

    if stats.complexity == "Simple" and object_count <= 2:
        return "Minimal Composition", CONF_DERIVED_STRONG
    if object_count <= 1:
        return "Centered", CONF_DERIVED_WEAK
    if stats.orientation == "Landscape" and object_count >= 4:
        return "Wide Shot", CONF_DERIVED_WEAK
    if stats.orientation == "Portrait":
        return "Portrait", CONF_DERIVED_WEAK
    return "Rule of Thirds", CONF_DERIVED_WEAK


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

# Styles that read as adjectives in a title. "Photorealistic Blue Sofa" is fine;
# "Photography Blue Sofa" is not, so non-adjectival mediums stay out of titles.
_TITLE_UNSUITABLE = {"Photography", "Digital Art", "Collage", "Sculpture"}


def generate_title(fields: dict) -> tuple[str, float]:
    """Build a descriptive, human-sounding title — never a random codename.

    Numbering (`#540`) is deliberately not applied here: collection indices
    belong to whatever mints the token, not to per-image analysis.
    """
    subject = fields.get("primary_subject") or ""
    scene = fields.get("scene") or ""

    # The noun is what the piece is *of*; a scene only wins when there is no
    # subject, otherwise it becomes a modifier.
    noun = subject or scene or fields.get("category") or "Composition"
    noun_words = {w.lower() for w in noun.split()}

    modifiers: list[str] = []

    def add(candidate: str | None) -> None:
        if not candidate or len(modifiers) >= 2:
            return
        if candidate in _TITLE_UNSUITABLE:
            return
        words = candidate.split()
        # Skip anything already implied by the noun ("Ceramic Ceramic Vase").
        if any(w.lower() in noun_words for w in words):
            return
        if len(words) > 2:
            return
        modifiers.append(candidate)
        noun_words.update(w.lower() for w in words)

    style = fields.get("style")
    colors = fields.get("dominant_colors") or []
    materials = fields.get("materials") or []

    add(style)
    # A colour is the single most recognisable hook in a title, so it outranks
    # material and mood for the second slot.
    add(colors[0] if colors else None)
    add(materials[0] if materials else None)
    add(fields.get("mood"))

    title = " ".join(modifiers + [noun])

    # Trim to something a marketplace card can display.
    words = title.split()
    if len(words) > 5:
        title = " ".join(words[:5])

    confidence = CONF_DERIVED_STRONG if modifiers and (subject or scene) else CONF_DERIVED_WEAK
    return title, confidence


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------

def _join_natural(items: list[str], limit: int = 4) -> str:
    items = [i.lower() for i in items[:limit]]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


# Vocabulary values that already read as a lighting phrase; appending the word
# "lighting" to them produces "studio lighting lighting".
_LIGHTING_SUFFIXES = ("light", "lighting", "daylight", "hour")


# Open, surface-like scenes read as "on a beach", enclosed ones as "in a kitchen".
_ON_SCENES = {"Beach", "Street", "Mountain", "Sky"}


def _scene_preposition(scene: str) -> str:
    return "on" if scene in _ON_SCENES else "in"


def _lighting_phrase(lighting: str) -> str:
    lowered = lighting.lower()
    return lowered if lowered.endswith(_LIGHTING_SUFFIXES) else f"{lowered} lighting"


# Each medium needs its own verb — "rendered as a oil painting" is not English.
_MEDIUM_PHRASES = {
    "3D Render": "Rendered in 3D.",
    "Photography": "Captured photographically.",
    "Pixel Art": "Created as pixel art.",
    "Digital Art": "Created as digital art.",
    "Vector Illustration": "Drawn as a vector illustration.",
    "Pencil Sketch": "Drawn as a pencil sketch.",
    "Ink Illustration": "Drawn in ink.",
    "Oil Painting": "Painted in oil.",
    "Watercolor": "Painted in watercolor.",
    "Acrylic Painting": "Painted in acrylic.",
    "Collage": "Assembled as a collage.",
    "Sculpture": "Presented as a sculpture.",
}


def generate_description(fields: dict) -> tuple[str, float]:
    """Assemble prose from observed fields only — no filler adjectives."""
    subject = (fields.get("primary_subject") or fields.get("scene") or "composition").lower()
    style = fields.get("style")
    materials = fields.get("materials") or []
    colors = fields.get("dominant_colors") or []
    scene = fields.get("scene")
    environment = fields.get("environment")
    objects = fields.get("objects") or []
    lighting = fields.get("lighting")
    mood = fields.get("mood")
    medium = fields.get("art_medium")

    # ── Sentence 1: what it is, in what style, what colour, where ──
    head_parts = []
    if style:
        head_parts.append(f"{style.lower()}-style")
    if materials:
        head_parts.append(materials[0].lower())

    lead = " ".join(head_parts)
    opener = f"{_article(lead or subject).capitalize()} {lead} {subject}".replace("  ", " ").strip()

    if colors:
        opener += f" in {_join_natural(colors, 3)} tones"

    if scene and scene.lower() not in subject:
        opener += f", set {_scene_preposition(scene)} {_article(scene)} {scene.lower()}"
    elif environment and not scene:
        opener += f", shown in {_article(environment)} {environment.lower()} setting"

    sentences = [opener + "."]

    # ── Sentence 2: the supporting cast ──
    supporting = _dedupe(
        [o for o in objects if o.lower() not in subject],
        exclude={subject},
    )
    if supporting:
        sentences.append(f"The scene includes {_join_natural(supporting, 4)}.")

    # ── Sentence 3: atmosphere ──
    if lighting and mood:
        phrase = _lighting_phrase(lighting)
        sentences.append(f"{phrase.capitalize()} gives the piece {_article(mood)} {mood.lower()} feel.")
    elif lighting:
        sentences.append(f"The piece is lit with {_lighting_phrase(lighting)}.")
    elif mood:
        sentences.append(f"The overall mood is {mood.lower()}.")

    if medium:
        sentences.append(
            _MEDIUM_PHRASES.get(medium, f"Created as {_article(medium)} {medium.lower()}.")
        )

    description = " ".join(sentences)

    # Confidence is capped by the fields the sentence actually leans on.
    contributors = [f for f in (style, subject, scene, mood) if f]
    confidence = CONF_DERIVED_STRONG if len(contributors) >= 3 else CONF_DERIVED_WEAK
    return description, confidence


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

# Padding only. Used when the model returned little or nothing, so that a
# listing is still discoverable — never at the expense of a specific tag.
_GENERIC_TAGS = [
    "nft", "digital-art", "collectible", "digital", "unique",
    "crypto-art", "original", "artwork", "blockchain", "one-of-one",
]


# Fragments that carry no search value on their own. Splitting "Off White" or
# "Wide Shot" into parts is useful for "white", useless for "off" and "shot".
_WEAK_FRAGMENTS = {
    "off", "set", "view", "shot", "angle", "level", "highly", "very", "hour",
    "composition", "art", "style", "quarter", "frame", "space", "front",
}


def generate_tags(fields: dict, stats: ImageStats) -> list[str]:
    """10-30 search tags, ordered most- to least-specific."""
    candidates: list[str] = []

    def push(value: object) -> None:
        if not value:
            return
        text = str(value)
        candidates.append(text)
        # Multi-word values also yield their parts: "Living Room" is worth
        # finding under "living-room" and "room" — but not under "off".
        words = text.split()
        if len(words) > 1:
            candidates.extend(
                w for w in words if len(w) >= 4 and w.lower() not in _WEAK_FRAGMENTS
            )

    push(fields.get("primary_subject"))
    push(fields.get("category"))
    push(fields.get("style"))
    push(fields.get("scene"))
    push(fields.get("art_medium"))
    # Colours come from the pixels, so they are available even when the model
    # contributed nothing at all.
    for value in (fields.get("dominant_colors") or stats.colors)[:3]:
        push(value)
    for value in (fields.get("materials") or [])[:3]:
        push(value)
    for value in (fields.get("secondary_subjects") or [])[:4]:
        push(value)
    for value in (fields.get("objects") or [])[:8]:
        push(value)
    push(fields.get("mood"))
    push(fields.get("lighting"))
    push(fields.get("environment"))
    push(fields.get("texture"))
    push(fields.get("pattern"))
    push(fields.get("perspective"))
    push(stats.complexity)
    push(stats.orientation)
    if stats.is_grayscale:
        push("monochrome")

    tags: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        slug = _slug(candidate)
        if len(slug) < 3 or slug in seen:
            continue
        seen.add(slug)
        tags.append(slug)

    for generic in _GENERIC_TAGS:
        if len(tags) >= 10:
            break
        if generic not in seen:
            seen.add(generic)
            tags.append(generic)

    return tags[:30]


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

# Scalar field -> marketplace trait name. Order is the display order.
_TRAIT_MAP = [
    ("category", "Category"),
    ("style", "Style"),
    ("art_medium", "Art Medium"),
    ("primary_subject", "Primary Subject"),
    ("scene", "Scene"),
    ("environment", "Environment"),
    ("lighting", "Lighting"),
    ("mood", "Mood"),
    ("composition", "Composition"),
    ("perspective", "Perspective"),
    ("texture", "Texture"),
    ("pattern", "Pattern"),
    ("visual_complexity", "Visual Complexity"),
]

_COLOR_TRAITS = ["Primary Color", "Secondary Color", "Accent Color"]
_MATERIAL_TRAITS = ["Primary Material", "Secondary Material"]


def generate_attributes(fields: dict, stats: ImageStats) -> list[Attribute]:
    """Every meaningful facet becomes a filterable trait — rarity analysis is
    only as good as the number of independent axes it can count."""
    attributes: list[Attribute] = []

    for key, trait_type in _TRAIT_MAP:
        value = fields.get(key)
        if value:
            attributes.append(Attribute(trait_type=trait_type, value=str(value)))

    for name, color in zip(_COLOR_TRAITS, fields.get("dominant_colors") or []):
        attributes.append(Attribute(trait_type=name, value=color))

    for name, material in zip(_MATERIAL_TRAITS, fields.get("materials") or []):
        attributes.append(Attribute(trait_type=name, value=material))

    attributes.append(Attribute(trait_type="Orientation", value=stats.orientation))

    if stats.is_grayscale:
        attributes.append(Attribute(trait_type="Palette", value="Monochrome"))

    # Numeric traits let marketplaces sort and bucket, not just filter.
    object_count = len(fields.get("objects") or [])
    if object_count:
        attributes.append(
            Attribute(trait_type="Object Count", value=object_count, display_type="number")
        )
    color_count = len(fields.get("dominant_colors") or [])
    if color_count:
        attributes.append(
            Attribute(trait_type="Color Count", value=color_count, display_type="number")
        )

    return attributes


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def generate_metadata(raw: dict, stats: ImageStats) -> NFTMetadata:
    fields, confidence = normalize_fields(raw)
    apply_image_stats(fields, confidence, stats)

    category, category_conf = infer_category(fields)
    if category:
        fields["category"] = category
        confidence["category"] = category_conf

    # Only derive composition if the model did not already provide one.
    if not fields.get("composition"):
        composition, composition_conf = infer_composition(fields, stats)
        fields["composition"] = composition
        confidence["composition"] = composition_conf

    title, title_conf = generate_title(fields)
    description, description_conf = generate_description(fields)

    confidence["title"] = title_conf
    confidence["description"] = description_conf

    tags = generate_tags(fields, stats)
    attributes = generate_attributes(fields, stats)

    # Tag and trait confidence is inherited: they can be no more reliable than
    # the fields they were assembled from.
    field_scores = [v for k, v in confidence.items() if k not in ("title", "description") and v > 0]
    inherited = round(sum(field_scores) / len(field_scores), 2) if field_scores else 0.0
    confidence["tags"] = inherited
    confidence["attributes"] = inherited

    return NFTMetadata(
        name=title,
        title=title,
        description=description,
        category=fields.get("category"),
        style=fields.get("style"),
        primary_subject=fields.get("primary_subject"),
        secondary_subjects=fields.get("secondary_subjects") or [],
        scene=fields.get("scene"),
        environment=fields.get("environment"),
        objects=fields.get("objects") or [],
        materials=fields.get("materials") or [],
        dominant_colors=fields.get("dominant_colors") or [],
        lighting=fields.get("lighting"),
        mood=fields.get("mood"),
        composition=fields.get("composition"),
        perspective=fields.get("perspective"),
        texture=fields.get("texture"),
        pattern=fields.get("pattern"),
        visual_complexity=fields.get("visual_complexity"),
        art_medium=fields.get("art_medium"),
        tags=tags,
        attributes=attributes,
        confidence={k: round(min(v, CEILING), 2) for k, v in sorted(confidence.items()) if v > 0},
    )
