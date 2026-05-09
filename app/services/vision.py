import json
import logging
import os
import threading
import time

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "vikhyatk/moondream2")
REVISION = os.getenv("MODEL_REVISION", "2025-01-09")

# Keep prompt minimal — fewer input tokens = faster
PROMPT = """Analyze this image. Return ONLY a JSON object:
{"object_type":"subject","style":"style","dominant_color":"color","mood":"mood","lighting":"lighting","environment":"environment"}"""

MAX_NEW_TOKENS = 150  # JSON response is ~100 tokens max


class VisionService:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._loaded = False
        self._device: torch.device | None = None

    def load_model(self) -> None:
        # ── Always use float16 to halve memory usage ──
        # CPU fp16 is slower than fp32 per-op but uses half the RAM,
        # which prevents OOM on small VPS instances.
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float16  # half memory on CPU

        self._device = device
        logger.info("Loading Moondream2 on %s (dtype=%s)...", device, dtype)

        cpu_threads = int(os.getenv("OMP_NUM_THREADS", "4"))
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
        logger.info("Torch threads: intra=%d inter=%d", cpu_threads, max(1, cpu_threads // 2))

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self._model.to(device)
        self._model.eval()

        self._loaded = True

        # Log memory footprint
        param_bytes = sum(p.nelement() * p.element_size() for p in self._model.parameters())
        logger.info("Moondream2 ready. Model size: %.0f MB", param_bytes / 1024 / 1024)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def analyze(self, image: Image.Image) -> dict:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Wait for startup to complete.")

        with self._lock, torch.inference_mode():
            t0 = time.monotonic()
            logger.info("Encoding image...")
            enc_image = self._model.encode_image(image)
            t1 = time.monotonic()
            logger.info("Image encoded (%.2fs). Generating answer (max %d tokens)...", t1 - t0, MAX_NEW_TOKENS)

            raw = self._model.answer_question(
                enc_image,
                PROMPT,
                self._tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            t2 = time.monotonic()
            logger.info("Answer generated (%.2fs). Raw output: %s", t2 - t1, raw[:200])

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
