import random

from PIL import Image, ImageDraw

from app.services.imagestats import (
    analyze_pixels,
    estimate_lighting,
    extract_palette,
    measure_complexity,
    name_color,
)


def _solid(color, size=(200, 200)) -> Image.Image:
    return Image.new("RGB", size, color=color)


class TestColorNaming:
    def test_primaries_get_expected_names(self):
        assert name_color((255, 0, 0)) == "Red"
        assert name_color((0, 0, 255)) == "Blue"
        assert name_color((255, 255, 255)) == "White"
        assert name_color((0, 0, 0)) == "Black"

    def test_light_and_dark_variants_are_distinguished(self):
        assert name_color((160, 200, 235)) == "Light Blue"
        assert name_color((25, 40, 90)) == "Navy"


class TestPalette:
    def test_solid_image_yields_one_color(self):
        colors, weights = extract_palette(_solid((200, 46, 46)))
        assert colors == ["Red"]
        assert weights[0] > 0.95

    def test_two_tone_image_yields_both_ordered_by_area(self):
        img = _solid((252, 252, 252))
        ImageDraw.Draw(img).rectangle([0, 0, 199, 139], fill=(46, 96, 200))
        colors, weights = extract_palette(img)
        assert colors[0] == "Blue"
        assert "White" in colors
        assert weights[0] > weights[1]

    def test_tiny_slivers_are_excluded(self):
        img = _solid((252, 252, 252))
        ImageDraw.Draw(img).rectangle([0, 0, 4, 4], fill=(200, 46, 46))
        colors, _ = extract_palette(img)
        assert "Red" not in colors

    def test_weights_are_shares_not_counts(self):
        _, weights = extract_palette(_solid((58, 150, 70)))
        assert all(0 <= w <= 1 for w in weights)


class TestComplexity:
    def test_flat_image_is_simple(self):
        label, score = measure_complexity(_solid((128, 128, 128)))
        assert label == "Simple"
        assert score < 0.06

    def test_noise_is_more_complex_than_flat(self):
        rng = random.Random(0)
        noise = Image.new("RGB", (200, 200))
        noise.putdata([(rng.randrange(256),) * 3 for _ in range(200 * 200)])
        _, noisy_score = measure_complexity(noise)
        _, flat_score = measure_complexity(_solid((128, 128, 128)))
        assert noisy_score > flat_score

    def test_label_is_from_the_controlled_vocabulary(self):
        label, _ = measure_complexity(_solid((10, 200, 10)))
        assert label in {"Simple", "Medium", "Complex", "Highly Detailed"}


class TestLighting:
    def test_dark_image_reads_as_night(self):
        label, conf, brightness, _, _ = estimate_lighting(_solid((12, 12, 14)))
        assert label == "Night"
        assert conf > 0
        assert brightness < 0.16

    def test_dim_image_reads_as_low_light(self):
        assert estimate_lighting(_solid((60, 60, 64)))[0] == "Low Light"

    def test_warm_cast_is_detected(self):
        assert estimate_lighting(_solid((235, 200, 140)))[0] == "Warm Light"

    def test_cool_cast_is_detected(self):
        assert estimate_lighting(_solid((150, 175, 215)))[0] == "Cool Light"

    def test_bright_flat_image_reads_as_soft(self):
        assert estimate_lighting(_solid((225, 225, 225)))[0] == "Soft Light"

    def test_confidence_stays_modest(self):
        # Brightness alone cannot prove a lighting setup; never claim it can.
        for rgb in [(12, 12, 14), (60, 60, 64), (235, 200, 140), (150, 175, 215)]:
            assert estimate_lighting(_solid(rgb))[1] <= 0.65


class TestSharedThumbnail:
    def test_small_image_is_not_copied(self):
        from app.services.imagestats import _thumbnail

        img = _solid((10, 20, 30), size=(64, 64))
        assert _thumbnail(img, 128) is img  # no wasted copy

    def test_large_image_is_downscaled(self):
        from app.services.imagestats import _thumbnail

        result = _thumbnail(_solid((10, 20, 30), size=(900, 600)), 128)
        assert max(result.size) <= 128


class TestAnalyzePixels:
    def test_orientation_from_geometry(self):
        assert analyze_pixels(_solid((100, 100, 100), (400, 200))).orientation == "Landscape"
        assert analyze_pixels(_solid((100, 100, 100), (200, 400))).orientation == "Portrait"
        assert analyze_pixels(_solid((100, 100, 100), (300, 300))).orientation == "Square"

    def test_grayscale_detection(self):
        assert analyze_pixels(_solid((120, 120, 120))).is_grayscale is True
        assert analyze_pixels(_solid((200, 46, 46))).is_grayscale is False

    def test_returns_complete_stats(self):
        stats = analyze_pixels(_solid((46, 96, 200), (400, 300)))
        assert stats.colors
        assert len(stats.colors) == len(stats.color_weights)
        assert stats.complexity
        assert stats.aspect_ratio > 1
