import io

from PIL import Image

from app.utils.image import preprocess_image


def _make_png_bytes(width: int = 100, height: int = 80, mode: str = "RGB") -> bytes:
    img = Image.new(mode, (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_resizes_to_512():
    raw = _make_png_bytes(1024, 768)
    result = preprocess_image(raw)
    assert result.size == (512, 512)


def test_preprocess_converts_rgba_to_rgb():
    raw = _make_png_bytes(64, 64, mode="RGBA")
    result = preprocess_image(raw)
    assert result.mode == "RGB"
