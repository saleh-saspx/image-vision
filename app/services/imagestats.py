"""Cheap, deterministic facts read straight from the pixels.

A 2B-parameter VLM is a poor colour meter and a poor complexity judge. Both are
measurable directly, in single-digit milliseconds, with far better accuracy than
a generated token — so we measure them instead of asking, and spend the model's
limited output budget on semantics only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image, ImageFilter, ImageStat

# Every analyser below works on a downscaled copy. Producing that copy once and
# sharing it is the single biggest win in this module: the four analysers used
# to each copy and resize the full-resolution image independently, doing the
# same expensive work four times over.
_BASE_SIZE = 128

# Named palette: NFT metadata is filtered by humans, so colour names must be the
# ones a creator would type ("Light Blue"), not "#A3C4E0" or "cornflower".
_NAMED_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("Black", (16, 16, 18)),
    ("Charcoal", (52, 54, 58)),
    ("Dark Gray", (88, 90, 94)),
    ("Gray", (132, 134, 138)),
    ("Light Gray", (188, 190, 194)),
    ("Off White", (238, 236, 230)),
    ("White", (252, 252, 252)),
    ("Cream", (245, 232, 202)),
    ("Beige", (222, 205, 172)),
    ("Tan", (198, 166, 120)),
    ("Brown", (128, 88, 52)),
    ("Dark Brown", (74, 50, 30)),
    ("Rust", (164, 76, 40)),
    ("Orange", (240, 130, 30)),
    ("Peach", (250, 190, 150)),
    ("Gold", (212, 170, 60)),
    ("Yellow", (244, 214, 60)),
    ("Olive", (128, 128, 52)),
    ("Lime", (170, 214, 60)),
    ("Green", (58, 150, 70)),
    ("Dark Green", (30, 84, 48)),
    ("Mint", (150, 224, 190)),
    ("Teal", (36, 148, 148)),
    ("Cyan", (70, 210, 220)),
    ("Light Blue", (150, 195, 232)),
    ("Sky Blue", (100, 176, 236)),
    ("Blue", (42, 84, 206)),
    # Fully saturated sRGB blue sits at the violet edge in Lab and would
    # otherwise name itself "Purple". A second anchor keeps it honest; anchors
    # sharing a name merge during palette extraction.
    ("Blue", (12, 12, 245)),
    ("Navy", (28, 46, 96)),
    ("Indigo", (75, 0, 130)),
    ("Purple", (150, 52, 200)),
    ("Lavender", (196, 176, 226)),
    ("Magenta", (206, 62, 172)),
    ("Pink", (238, 140, 178)),
    ("Rose", (222, 96, 118)),
    ("Red", (200, 46, 46)),
    ("Dark Red", (128, 28, 32)),
    ("Maroon", (104, 40, 58)),
    ("Silver", (198, 202, 208)),
]


@dataclass
class ImageStats:
    """Everything the pipeline can know without running the model."""

    colors: list[str] = field(default_factory=list)
    color_weights: list[float] = field(default_factory=list)
    complexity: str = "Medium"
    complexity_score: float = 0.0
    orientation: str = "Square"
    aspect_ratio: float = 1.0
    is_grayscale: bool = False
    # Lighting read from the pixels. Only used when the model did not name it —
    # a fallback, never an override.
    lighting: str | None = None
    lighting_confidence: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    warmth: float = 0.0


def _thumbnail(image: Image.Image, size: int) -> Image.Image:
    """Downscale to fit `size`, or hand back the original when already small.

    Skipping the copy for images that are already below the target is what lets
    analyze_pixels share one base thumbnail across every analyser.
    """
    if max(image.size) <= size:
        return image
    thumb = image.copy()
    thumb.thumbnail((size, size), Image.BILINEAR)
    return thumb


def _srgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """sRGB -> CIELAB (D65). Nearest-colour naming in RGB space matches human
    judgement badly; Lab is close enough to perceptual to pick the right name."""

    def to_linear(channel: int) -> float:
        c = channel / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (to_linear(c) for c in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


_NAMED_LAB = [(name, _srgb_to_lab(rgb)) for name, rgb in _NAMED_COLORS]

# Achromatic names. They sit in the middle of Lab space and therefore act as
# attractors: a warm off-white wall lands on "Off White" and a pale blue sofa
# on "Light Gray", which is exactly the flat, useless palette a creator would
# delete. Once a colour carries visible tint we drop these from the running and
# force a name that says something ("Cream", "Light Blue").
_NEUTRAL_NAMES = {
    "Black", "Charcoal", "Dark Gray", "Gray", "Light Gray",
    "White", "Off White", "Silver",
}

# Lab chroma above which a colour is no longer meaningfully neutral. Low enough
# to catch subtle tints, high enough that genuine greys stay grey.
_NEUTRAL_CHROMA_MAX = 7.0

_CHROMATIC_LAB = [(name, lab) for name, lab in _NAMED_LAB if name not in _NEUTRAL_NAMES]


def name_color(rgb: tuple[int, int, int]) -> str:
    lab = _srgb_to_lab(rgb)
    chroma = (lab[1] ** 2 + lab[2] ** 2) ** 0.5

    candidates = _NAMED_LAB if chroma <= _NEUTRAL_CHROMA_MAX else _CHROMATIC_LAB
    return min(
        candidates,
        key=lambda item: sum((a - b) ** 2 for a, b in zip(lab, item[1])),
    )[0]


def extract_palette(image: Image.Image, max_colors: int = 5) -> tuple[list[str], list[float]]:
    """Ordered dominant colour names plus each one's share of the image.

    Quantising a 96px thumbnail is ~1ms and loses nothing: dominant colour is a
    low-frequency property, so full resolution buys no accuracy.
    """
    thumb = _thumbnail(image, 96)

    quantized = thumb.quantize(colors=16, method=Image.MEDIANCUT, dither=Image.NONE)
    palette = quantized.getpalette() or []
    counts = quantized.getcolors() or []
    total = sum(count for count, _ in counts) or 1

    # Merge quantiser bins that collapse onto the same human colour name —
    # eight shades of "Blue" is not a useful palette for a creator.
    merged: dict[str, float] = {}
    for count, index in sorted(counts, reverse=True):
        rgb = tuple(palette[index * 3: index * 3 + 3])
        if len(rgb) != 3:
            continue
        name = name_color(rgb)
        merged[name] = merged.get(name, 0.0) + count / total

    ordered = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    # Drop slivers; a colour under 4% of the frame is noise, not a trait.
    ordered = [(name, share) for name, share in ordered if share >= 0.04][:max_colors]

    return [name for name, _ in ordered], [round(share, 4) for _, share in ordered]


def measure_complexity(image: Image.Image) -> tuple[str, float]:
    """Edge density as a proxy for visual busyness.

    A flat vector logo and a detailed oil painting differ enormously in edge
    energy, which tracks what creators mean by "Highly Detailed" better than
    any label the model would guess.
    """
    thumb = _thumbnail(image, 128).convert("L")

    edges = thumb.filter(ImageFilter.FIND_EDGES)
    histogram = edges.histogram()
    pixels = sum(histogram) or 1
    # Share of pixels carrying meaningful edge energy.
    score = sum(histogram[24:]) / pixels

    if score < 0.06:
        label = "Simple"
    elif score < 0.18:
        label = "Medium"
    elif score < 0.34:
        label = "Complex"
    else:
        label = "Highly Detailed"

    return label, round(score, 4)


def estimate_lighting(image: Image.Image) -> tuple[str | None, float, float, float, float]:
    """Infer lighting from luminance and colour temperature.

    A dark frame is obviously low-light and a warm-cast frame is obviously warm
    lit — returning null for those, as the previous version did, made the
    creator fill in something the pixels already stated. Confidence stays
    modest because brightness alone cannot distinguish, say, backlit from
    overcast; the model's own answer always wins when it gives one.

    Returns (lighting, confidence, brightness, contrast, warmth).
    """
    thumb = _thumbnail(image, 64)

    # ImageStat runs in C. The equivalent Python loop over even a 64px
    # thumbnail was the slowest thing in this module.
    gray_stat = ImageStat.Stat(thumb.convert("L"))
    brightness = gray_stat.mean[0] / 255.0
    contrast = gray_stat.stddev[0] / 255.0

    # Mean b* in Lab: positive is yellow (warm), negative is blue (cool).
    # One conversion of the average colour, not one per pixel.
    rgb_stat = ImageStat.Stat(thumb)
    average = tuple(int(c) for c in rgb_stat.mean[:3])
    warmth = _srgb_to_lab(average)[2]

    if brightness < 0.16:
        lighting, confidence = "Night", 0.62
    elif brightness < 0.30:
        lighting, confidence = "Low Light", 0.58
    elif brightness > 0.80 and contrast < 0.14:
        # Bright and flat is the signature of a lit studio backdrop.
        lighting, confidence = "Soft Light", 0.50
    elif warmth > 14:
        lighting, confidence = "Warm Light", 0.52
    elif warmth < -8:
        lighting, confidence = "Cool Light", 0.52
    elif brightness > 0.55:
        lighting, confidence = "Natural Daylight", 0.45
    else:
        lighting, confidence = None, 0.0

    return lighting, confidence, round(brightness, 4), round(contrast, 4), round(warmth, 2)


def _orientation(width: int, height: int) -> tuple[str, float]:
    ratio = width / height if height else 1.0
    if 0.95 <= ratio <= 1.05:
        return "Square", ratio
    return ("Landscape" if ratio > 1 else "Portrait"), ratio


def _is_grayscale(image: Image.Image) -> bool:
    thumb = _thumbnail(image, 48)
    return all(
        max(pixel[:3]) - min(pixel[:3]) <= 12
        for pixel in thumb.getdata()
    )


def analyze_pixels(image: Image.Image) -> ImageStats:
    """Single pass over a thumbnail; runs before the model, costs ~5ms."""
    # One downscale, shared by every analyser below. Orientation is read from
    # the original dimensions, since the thumbnail preserves aspect ratio but
    # not size.
    orientation, ratio = _orientation(*image.size)
    base = _thumbnail(image, _BASE_SIZE)

    colors, weights = extract_palette(base)
    complexity, score = measure_complexity(base)
    lighting, lighting_confidence, brightness, contrast, warmth = estimate_lighting(base)

    return ImageStats(
        colors=colors,
        color_weights=weights,
        complexity=complexity,
        complexity_score=score,
        orientation=orientation,
        aspect_ratio=round(ratio, 3),
        is_grayscale=_is_grayscale(base),
        lighting=lighting,
        lighting_confidence=lighting_confidence,
        brightness=brightness,
        contrast=contrast,
        warmth=warmth,
    )
