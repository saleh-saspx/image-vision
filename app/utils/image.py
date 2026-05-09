from PIL import Image
import io

TARGET_SIZE = (512, 512)


def preprocess_image(raw_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw_bytes))
    img = img.convert("RGB")
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return img
