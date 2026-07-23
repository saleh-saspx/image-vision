"""Canonical vocabularies for NFT metadata fields.

Free-form model output is mapped onto a controlled vocabulary so that
marketplace filtering, rarity analysis and search actually work — "neon-lit",
"neon lighting" and "Neon" must all collapse to one facet value.

The match tier is also the honest signal for confidence: an exact vocabulary
hit means the model said something we recognise, a fuzzy hit means we guessed,
and a passthrough means we are just echoing the model.
"""
from __future__ import annotations

import re

# Match quality tiers, ordered strongest first.
MATCH_EXACT = "exact"
MATCH_FUZZY = "fuzzy"
MATCH_PASSTHROUGH = "passthrough"
MATCH_NONE = "none"

_NULL_VALUES = {
    "", "unknown", "none", "n/a", "na", "null", "nil", "not sure", "unclear",
    "undefined", "no", "nothing", "not applicable", "-",
}

# A small VLM frequently echoes the schema back instead of describing the image:
# asked for "main subject" it answers "main subject". These reach the creator as
# "Gray Main Subject" or a tag called `object`, which is worse than an empty
# field — the whole point is metadata a human would accept without editing.
# Every one of these is dropped as if the model had said nothing.
_PLACEHOLDER_VALUES = {
    "subject", "subjects", "main subject", "primary subject", "other subject",
    "other subjects", "secondary subject", "secondary subjects",
    "object", "objects", "generic object", "visible object", "visible objects",
    "item", "items", "thing", "things", "stuff", "element", "elements",
    "artwork", "art work", "art", "image", "picture", "scene", "place",
    "setting", "background", "foreground", "content", "description", "title",
    "style", "color", "colour", "colors", "colours", "material", "materials",
    # "medium" is absent here on purpose — see _FIELD_PLACEHOLDERS below.
    "mood", "lighting", "texture", "pattern", "category",
    "various", "several", "multiple", "other", "misc", "miscellaneous",
    "n a", "tbd", "todo", "example", "sample", "placeholder", "text", "value",
}

# Some words are a placeholder in one field and a real answer in another.
# "Medium" is the canonical visual_complexity value, but as an art_medium it is
# just the schema echoed back, so the check has to know which field it is in.
_FIELD_PLACEHOLDERS: dict[str, set[str]] = {
    "art_medium": {"medium", "art medium", "mediums"},
    "visual_complexity": {"complexity", "visual complexity"},
    "primary_subject": {"main", "primary", "focus", "centerpiece"},
    "environment": {"environment", "env"},
    "composition": {"composition"},
    "perspective": {"perspective", "angle", "view", "camera angle"},
}

# The same schema-echo wearing qualifiers or an index: "Other Notable
# Subjects", "Generic Object", "Visible Objects", "Object 1", "Item #2".
# Enumerating the exact phrases does not scale — the model recombines these
# words freely — so the shape is matched instead: any number of vague
# qualifiers followed by a vague noun, optionally numbered.
_PLACEHOLDER_RE = re.compile(
    r"^(?:(?:other|another|additional|more|secondary|primary|main|generic|"
    r"visible|notable|important|key|various|several|misc|unnamed|unidentified)\s+)*"
    r"(?:object|item|subject|thing|element|artwork|image|entity|figure|detail|"
    r"feature|piece|content)s?"
    r"(?:\s*#?\s*(?:\d+|[a-z]))?$"
)

# ---------------------------------------------------------------------------
# Vocabularies: canonical value -> extra surface forms that map onto it.
# The canonical value itself is always matched, no need to repeat it.
# ---------------------------------------------------------------------------

STYLE = {
    "Classic": ["classical", "traditional", "timeless"],
    "Modern": ["contemporary", "mid-century", "mid century modern"],
    "Minimalist": ["minimal", "minimalistic", "simple", "clean"],
    "Industrial": ["loft", "warehouse", "raw industrial"],
    "Victorian": ["baroque", "rococo", "ornate", "antique"],
    "Cyberpunk": ["cyber", "neon futuristic", "cyber punk", "dystopian"],
    "Steampunk": ["steam punk", "clockwork", "brass punk"],
    "Fantasy": ["mythical", "magical", "fairytale", "medieval fantasy"],
    "Sci-Fi": ["science fiction", "scifi", "futuristic", "space age"],
    "Anime": ["manga", "japanese animation", "cel shaded"],
    "Cartoon": ["comic", "toon", "illustrated cartoon"],
    "Pixel Art": ["pixel", "8-bit", "8 bit", "16-bit", "retro pixel"],
    "Realistic": ["real", "lifelike", "naturalistic"],
    "Photorealistic": ["photo realistic", "hyperrealistic", "hyper realistic"],
    "Surreal": ["surrealist", "dreamlike", "dreamy"],
    "Abstract": ["non-representational", "nonrepresentational"],
    "Vaporwave": ["vapor wave", "synthwave", "retrowave", "outrun"],
    "Gothic": ["goth", "dark gothic"],
    "Bohemian": ["boho", "eclectic"],
    "Scandinavian": ["nordic", "scandi"],
    "Luxury": ["luxurious", "opulent", "high end", "premium", "lavish"],
    "Rustic": ["farmhouse", "country style", "rural", "weathered wood"],
    "Art Deco": ["deco", "art-deco"],
    "Grunge": ["distressed", "gritty"],
    "Low Poly": ["lowpoly", "low-poly", "polygonal"],
    "Vintage": ["retro", "old fashioned", "nostalgic"],
}

ART_MEDIUM = {
    "Digital Art": ["digital", "digital painting", "digital illustration", "cg"],
    "3D Render": ["3d", "3d rendering", "cgi", "octane render", "blender render"],
    "Photography": ["photo", "photograph", "photographic", "camera"],
    "Oil Painting": ["oil", "oil on canvas", "painting"],
    "Watercolor": ["water color", "watercolour", "aquarelle"],
    "Acrylic Painting": ["acrylic"],
    "Pencil Sketch": ["pencil", "sketch", "graphite", "drawing"],
    "Ink Illustration": ["ink", "pen and ink", "line art", "lineart"],
    "Vector Illustration": ["vector", "flat illustration", "svg"],
    "Pixel Art": ["pixel", "sprite", "8-bit art"],
    "Collage": ["mixed media", "photomontage"],
    "Sculpture": ["sculpted", "statue", "carving"],
}

SCENE = {
    "Living Room": ["lounge", "sitting room", "family room", "livingroom"],
    "Bedroom": ["bed room", "sleeping room"],
    "Kitchen": ["kitchenette", "cooking area"],
    "Dining Room": ["dining area", "dining space"],
    "Bathroom": ["bath room", "washroom", "restroom"],
    "Office": ["workspace", "study", "home office", "desk setup"],
    "Studio": ["photo studio", "studio backdrop", "seamless backdrop"],
    "Street": ["city street", "alley", "sidewalk", "road"],
    "City": ["cityscape", "downtown", "urban skyline", "skyline"],
    "Beach": ["seaside", "shore", "coast", "coastline"],
    "Ocean": ["sea", "underwater", "open water"],
    "Mountain": ["mountains", "peak", "alpine", "highlands"],
    "Forest": ["woods", "woodland", "jungle", "trees"],
    "Desert": ["dunes", "sand dunes", "arid"],
    "Garden": ["backyard", "park", "greenhouse"],
    "Space": ["outer space", "cosmos", "galaxy", "nebula", "universe"],
    "Sky": ["clouds", "cloudscape", "aerial"],
    "Cafe": ["coffee shop", "restaurant", "bar"],
    "Museum": ["gallery", "exhibition"],
    "Temple": ["church", "cathedral", "shrine", "ruins"],
    "Laboratory": ["lab", "workshop"],
    "Void": ["empty background", "plain background", "blank background", "no background"],
}

ENVIRONMENT = {
    "Indoor": ["inside", "interior", "indoors"],
    "Outdoor": ["outside", "exterior", "outdoors", "open air"],
    "Studio": ["studio setting", "seamless background"],
    "Natural": ["nature", "wilderness", "natural setting"],
    "Urban": ["city", "metropolitan", "downtown"],
    "Underwater": ["submerged", "beneath the sea"],
    "Outer Space": ["space", "cosmic", "orbit"],
    "Fantasy Realm": ["fantasy world", "magical realm", "otherworldly"],
    "Abstract Space": ["abstract background", "no environment", "undefined space"],
}

LIGHTING = {
    "Natural Daylight": ["daylight", "natural light", "sunlight", "sunny", "day", "bright daylight"],
    "Golden Hour": ["sunset", "sunrise", "dusk", "dawn", "warm sunset"],
    "Studio Lighting": ["studio", "softbox", "controlled lighting"],
    "Soft Light": ["soft", "diffused", "diffuse", "gentle light"],
    "Warm Light": ["warm", "warm tones", "warm glow"],
    "Cool Light": ["cool", "cool tones", "blue light"],
    "Backlit": ["back lit", "rim light", "silhouette lighting", "rim lighting"],
    "Low Light": ["dim", "dim light", "moody lighting", "shadowy"],
    "Night": ["nighttime", "night time", "dark", "moonlight"],
    "Neon": ["neon lights", "neon glow", "neon lighting"],
    "Dramatic": ["high contrast", "chiaroscuro", "harsh"],
    "Ambient": ["even lighting", "flat lighting", "ambient light"],
}

MOOD = {
    "Luxury": ["luxurious", "opulent", "lavish", "premium", "rich"],
    "Elegant": ["elegance", "refined", "sophisticated", "graceful"],
    "Minimal": ["understated", "restrained", "clean and simple"],
    "Calm": ["peaceful", "tranquil", "relaxed", "quiet"],
    "Serene": ["serenity", "meditative", "still"],
    "Cozy": ["warm and inviting", "homely", "comfortable", "inviting", "homey"],
    "Dark": ["gloomy", "somber", "ominous", "grim"],
    "Mysterious": ["mystical", "enigmatic", "mystery", "eerie"],
    "Energetic": ["vibrant", "dynamic", "lively", "bold", "playful"],
    "Nostalgic": ["vintage feel", "wistful", "retro mood"],
    "Professional": ["corporate", "business", "clinical"],
    "Dreamy": ["ethereal", "whimsical", "soft dreamy"],
    "Melancholic": ["sad", "lonely", "somber mood", "melancholy"],
    "Futuristic": ["hi-tech", "high tech", "advanced"],
}

COMPOSITION = {
    "Centered": ["center", "central", "symmetrical", "symmetry", "middle"],
    "Rule of Thirds": ["thirds", "off center", "off-center"],
    "Wide Shot": ["wide", "wide angle", "establishing shot", "full shot"],
    "Close Up": ["closeup", "close-up", "macro", "detail shot"],
    "Minimal Composition": ["minimalist composition", "negative space", "sparse"],
    "Layered": ["depth", "foreground and background", "layered depth"],
    "Diagonal": ["dynamic angle", "tilted"],
    "Full Frame": ["fills the frame", "edge to edge"],
}

PERSPECTIVE = {
    "Front View": ["frontal", "front", "head on", "head-on", "straight on"],
    "Side View": ["side", "profile", "lateral"],
    "Three-Quarter View": ["three quarter", "3/4 view", "angled view"],
    "Top View": ["top down", "top-down", "overhead", "birds eye", "bird's eye", "flat lay"],
    "Isometric": ["iso", "isometric view", "axonometric"],
    "Eye Level": ["eye-level", "neutral angle", "straight ahead"],
    "Low Angle": ["from below", "worms eye", "worm's eye", "looking up"],
    "High Angle": ["from above", "looking down"],
    "Aerial View": ["drone shot", "aerial"],
}

TEXTURE = {
    "Smooth": ["sleek", "even", "polished"],
    "Rough": ["coarse", "gritty", "textured"],
    "Glossy": ["shiny", "shine", "gloss", "lacquered"],
    "Matte": ["flat finish", "non reflective", "dull"],
    "Soft": ["plush", "fuzzy", "velvety", "fluffy"],
    "Reflective": ["mirror", "mirrored", "specular", "chrome"],
    "Grainy": ["noisy", "film grain", "grain"],
    "Weathered": ["worn", "aged", "distressed", "rusted", "rusty"],
}

PATTERN = {
    "Plain": ["solid", "none", "no pattern", "uniform", "unpatterned"],
    "Striped": ["stripes", "lines", "linear pattern"],
    "Geometric": ["geometry", "shapes", "angular pattern", "grid"],
    "Floral": ["flowers", "botanical", "flower pattern"],
    "Abstract": ["organic pattern", "irregular"],
    "Checkered": ["checkerboard", "checked", "plaid", "tartan"],
    "Dotted": ["polka dot", "polka dots", "spotted", "dots"],
    "Gradient": ["ombre", "colour gradient", "color gradient"],
    "Repeating": ["tiled", "tessellated", "seamless pattern"],
}

MATERIAL = {
    "Wood": ["wooden", "timber", "oak", "walnut", "pine", "mahogany"],
    "Fabric": ["textile", "cloth", "upholstery", "cotton", "linen", "velvet", "wool"],
    "Leather": ["suede", "hide"],
    "Metal": ["metallic", "steel", "iron", "aluminum", "aluminium", "brass", "copper", "chrome"],
    "Gold": ["golden", "gilded", "gilt"],
    "Silver": ["silvery", "platinum"],
    "Glass": ["transparent glass", "glassy"],
    # Distinct from Glass on purpose: "Crystal Dragon" and "Crystal Vase" are
    # real NFT subjects, and collapsing them into "Glass" loses the trait.
    "Crystal": ["crystalline", "quartz", "gemstone", "diamond"],
    "Stone": ["marble", "granite", "rock", "concrete", "cement"],
    "Plastic": ["acrylic plastic", "vinyl", "resin"],
    "Paper": ["cardboard", "parchment"],
    "Ceramic": ["porcelain", "china"],
    "Clay": ["terracotta", "earthenware"],
    "Rubber": ["silicone"],
    "Neon": ["neon tube", "neon light"],
}

CATEGORY = {
    "Furniture": ["interior furniture", "home furniture"],
    "Photography": ["photo", "photograph"],
    "Digital Art": ["digital", "cg art", "3d art", "pixel art", "generative art"],
    "Painting": ["painted", "fine art", "canvas"],
    "Illustration": ["illustrated", "drawing", "comic art"],
    "Architecture": ["building", "architectural"],
    "Interior Design": ["interior", "interior scene", "home decor", "decor", "interiors"],
    "Landscape": ["scenery", "nature scene", "vista"],
    "Portrait": ["headshot", "person portrait", "face"],
    "Fashion": ["clothing", "apparel", "outfit"],
    "Gaming": ["game", "game asset", "video game"],
    "Fantasy": ["mythical", "magic"],
    "Collectible": ["collectable", "pfp", "avatar"],
    "Abstract": ["non representational"],
    "Vehicle": ["car", "automotive", "transport", "ship", "aircraft"],
    "Nature": ["botanical", "plants"],
    "Animal": ["creature", "pet", "wildlife"],
    "Technology": ["tech", "electronics", "gadget", "device"],
    "Anime": ["manga", "anime art"],
}

VISUAL_COMPLEXITY = {
    "Simple": ["basic", "sparse", "clean"],
    "Medium": ["moderate", "balanced", "average"],
    "Complex": ["busy", "intricate", "detailed"],
    "Highly Detailed": ["very detailed", "extremely detailed", "hyper detailed", "ornate"],
}


def _build_lookup(vocab: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, synonyms in vocab.items():
        lookup[canonical.lower()] = canonical
        for syn in synonyms:
            lookup[syn.lower()] = canonical
    return lookup


_LOOKUPS = {
    "style": _build_lookup(STYLE),
    "art_medium": _build_lookup(ART_MEDIUM),
    "scene": _build_lookup(SCENE),
    "environment": _build_lookup(ENVIRONMENT),
    "lighting": _build_lookup(LIGHTING),
    "mood": _build_lookup(MOOD),
    "composition": _build_lookup(COMPOSITION),
    "perspective": _build_lookup(PERSPECTIVE),
    "texture": _build_lookup(TEXTURE),
    "pattern": _build_lookup(PATTERN),
    "material": _build_lookup(MATERIAL),
    "category": _build_lookup(CATEGORY),
    "visual_complexity": _build_lookup(VISUAL_COMPLEXITY),
}

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {"a", "an", "the", "of", "with", "and", "in", "on", "is", "are", "it", "this", "that", "very"}


def clean_text(value: object) -> str:
    """Normalise raw model output into a comparable lowercase string."""
    if value is None:
        return ""
    text = str(value).strip().strip("\"'.,;:").lower()
    text = re.sub(r"\s+", " ", text)

    # An uncopied prompt slot, e.g. "<room or place>". Always a non-answer.
    if text.startswith("<") or text.endswith(">"):
        return ""
    if text in _NULL_VALUES or text in _PLACEHOLDER_VALUES:
        return ""
    if _PLACEHOLDER_RE.match(text):
        return ""
    return text


def clean_open_text(field: str, value: object) -> str:
    """clean_text for open-vocabulary fields, honouring per-field placeholders."""
    text = clean_text(value)
    return "" if text in _FIELD_PLACEHOLDERS.get(field, ()) else text


def is_placeholder(value: object) -> bool:
    """True when the model echoed the schema instead of describing the image."""
    return not clean_text(value)


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text) if w not in _STOPWORDS}


def normalize(field: str, value: object) -> tuple[str | None, str]:
    """Map a raw value onto the canonical vocabulary for `field`.

    Returns (canonical_value_or_None, match_tier).
    """
    text = clean_text(value)
    if not text or text in _FIELD_PLACEHOLDERS.get(field, ()):
        return None, MATCH_NONE

    lookup = _LOOKUPS.get(field)
    if lookup is None:
        return _titleize(text), MATCH_PASSTHROUGH

    if text in lookup:
        return lookup[text], MATCH_EXACT

    # Substring containment, longest surface form wins ("soft neon glow" -> Neon).
    contained = [surface for surface in lookup if len(surface) > 3 and surface in text]
    if contained:
        return lookup[max(contained, key=len)], MATCH_FUZZY

    # Token overlap for reordered phrasing ("lighting: natural" -> Natural Daylight).
    value_tokens = _tokens(text)
    if value_tokens:
        best, best_score = None, 0.0
        for surface, canonical in lookup.items():
            surface_tokens = _tokens(surface)
            if not surface_tokens:
                continue
            overlap = len(value_tokens & surface_tokens) / len(surface_tokens)
            if overlap > best_score:
                best, best_score = canonical, overlap
        if best is not None and best_score >= 0.99:
            return best, MATCH_FUZZY

    return _titleize(text), MATCH_PASSTHROUGH


def _titleize(text: str) -> str:
    """Title-case free text without mangling short joining words."""
    small = {"of", "in", "on", "and", "with", "the", "a", "an"}
    words = text.split()
    out = []
    for i, word in enumerate(words):
        out.append(word if i and word in small else word.capitalize())
    return " ".join(out)


def canonical_values(field: str) -> list[str]:
    lookup = _LOOKUPS.get(field, {})
    return sorted(set(lookup.values()))
