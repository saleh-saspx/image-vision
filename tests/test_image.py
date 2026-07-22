import io

from PIL import Image

from app.utils.image import MAX_SIDE, preprocess_image


def _png_bytes(width: int = 100, height: int = 80, mode: str = "RGB", color="red") -> bytes:
    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_large_image_is_bounded_by_max_side():
    result = preprocess_image(_png_bytes(2048, 1536))
    assert max(result.size) <= MAX_SIDE


def test_aspect_ratio_is_preserved():
    result = preprocess_image(_png_bytes(1600, 800))
    width, height = result.size
    assert abs((width / height) - 2.0) < 0.02


def test_small_image_is_not_upscaled():
    result = preprocess_image(_png_bytes(64, 48))
    assert result.size == (64, 48)


def test_converts_rgba_to_rgb():
    result = preprocess_image(_png_bytes(64, 64, mode="RGBA"))
    assert result.mode == "RGB"


def test_converts_grayscale_to_rgb():
    result = preprocess_image(_png_bytes(64, 64, mode="L", color=128))
    assert result.mode == "RGB"
