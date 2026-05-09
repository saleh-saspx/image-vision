import io
import os

from PIL import Image

_size = int(os.getenv("IMAGE_TARGET_SIZE", "384"))
TARGET_SIZE = (_size, _size)


def preprocess_image(raw_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw_bytes))
    img = img.convert("RGB")
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return img
