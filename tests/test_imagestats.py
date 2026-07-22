import random

from PIL import Image, ImageDraw

from app.services.imagestats import analyze_pixels, extract_palette, measure_complexity, name_color


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
