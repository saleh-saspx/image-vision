import json
import logging
import threading

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-01-09"

PROMPT = """Analyze this image and extract visual attributes. Return ONLY a valid JSON object with these exact keys:
{
  "object_type": "main subject of the image",
  "style": "artistic style (e.g. Cyberpunk, Minimalist, Abstract, Realistic, Surreal, Pixel Art, Watercolor)",
  "dominant_color": "single dominant color name",
  "mood": "emotional mood (e.g. Dark, Serene, Energetic, Mysterious, Joyful)",
  "lighting": "lighting description (e.g. Neon, Natural, Dramatic, Soft, Backlit)",
  "environment": "background/setting description"
}
Return ONLY the JSON object, no other text."""


def _select_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


class VisionService:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._loaded = False
        self._device: torch.device | None = None

    def load_model(self) -> None:
        device, dtype = _select_device()
        self._device = device
        logger.info("Loading Moondream2 model on %s (dtype=%s)...", device, dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self._model.to(device)
        self._model.eval()
        self._loaded = True
        logger.info("Moondream2 model loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def analyze(self, image: Image.Image) -> dict:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Wait for startup to complete.")

        with self._lock, torch.inference_mode():
            enc_image = self._model.encode_image(image)
            raw = self._model.answer_question(enc_image, PROMPT, self._tokenizer)

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Robust JSON extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(raw: str) -> dict:
        raw = raw.strip()

        # Strip markdown fences if model wraps output
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fall back: extract first JSON object from noisy output
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse model output as JSON: {raw[:300]}")
            else:
                raise ValueError(f"No JSON object found in model output: {raw[:300]}")

        required = {"object_type", "style", "dominant_color", "mood", "lighting", "environment"}
        for key in required:
            if key not in data:
                data[key] = "Unknown"
            else:
                data[key] = str(data[key]).strip() or "Unknown"

        return data


vision_service = VisionService()
