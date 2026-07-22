import io
import os

from PIL import Image, ImageOps

MAX_SIDE = int(os.getenv("IMAGE_TARGET_SIZE", "384"))


def preprocess_image(raw_bytes: bytes) -> Image.Image:
    """Decode once, downscale to a bounded box, preserve aspect ratio.

    The previous square resize distorted every non-square upload, which corrupts
    exactly the cues the model reads for perspective and composition. Bounding
    the longest side instead keeps geometry intact at the same pixel cost.
    """
    img = Image.open(io.BytesIO(raw_bytes))

    # draft() lets the JPEG decoder downscale during decode instead of after,
    # so a large photo never materialises at full resolution. No-op for PNG.
    img.draft("RGB", (MAX_SIDE, MAX_SIDE))

    # Honour EXIF rotation before anything reads orientation from the pixels.
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    if max(img.size) > MAX_SIDE:
        # BILINEAR over LANCZOS: at these ratios the model cannot tell the
        # difference, and it is several times cheaper per megapixel.
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)

    return img
