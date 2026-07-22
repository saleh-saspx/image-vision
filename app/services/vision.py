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

# Single-pass semantic extraction.
#
# Everything measurable from pixels (colours, complexity, orientation) is
# deliberately absent — it is computed in imagestats.py for free. The model's
# token budget goes entirely to semantics it alone can provide.
#
# Short keys are not cosmetic: on CPU, decode is the dominant cost and is
# strictly sequential, so every key character is latency. Abbreviating the
# schema saves ~40 output tokens per request versus spelled-out field names.
PROMPT = (
    "Describe this image. Reply with ONLY this JSON, no other text:\n"
    '{"subj":"main subject","sec":["other notable subjects"],'
    '"obj":["every visible object"],"scene":"place or setting",'
    '"env":"indoor or outdoor","sty":"art or design style",'
    '"med":"medium: photo, 3d render, painting, pixel art, digital art",'
    '"mat":["materials"],"lit":"lighting","mood":"mood",'
    '"persp":"camera angle","tex":"surface texture","pat":"pattern"}'
)

# Ceiling only — generation stops at the closing brace via EOS in practice.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# Raw model key -> pipeline field name.
KEY_MAP = {
    "subj": "primary_subject",
    "sec": "secondary_subjects",
    "obj": "objects",
    "scene": "scene",
    "env": "environment",
    "sty": "style",
    "med": "art_medium",
    "mat": "materials",
    "lit": "lighting",
    "mood": "mood",
    "persp": "perspective",
    "tex": "texture",
    "pat": "pattern",
}

LIST_FIELDS = {"secondary_subjects", "objects", "materials"}


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

        self._warmup()

    def _warmup(self) -> None:
        """Force lazy kernel/allocator init so the first real upload is not
        3-5x slower than steady state."""
        try:
            t0 = time.monotonic()
            dummy = Image.new("RGB", (256, 256), color=(127, 127, 127))
            with torch.inference_mode():
                enc = self._model.encode_image(dummy)
                self._model.answer_question(enc, "Describe.", self._tokenizer, max_new_tokens=4)
            logger.info("Warmup complete (%.2fs)", time.monotonic() - t0)
        except Exception:
            # A failed warmup must never block startup — it is pure optimisation.
            logger.warning("Warmup failed; first request will be slower", exc_info=True)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def analyze(self, image: Image.Image) -> dict:
        """One image encode, one generation pass. Returns raw semantic fields."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Wait for startup to complete.")

        with self._lock, torch.inference_mode():
            t0 = time.monotonic()
            # The encode is the expensive half on CPU; it happens exactly once
            # and the embedding is reused for any follow-up query below.
            enc_image = self._model.encode_image(image)
            t1 = time.monotonic()
            logger.info("Image encoded (%.2fs). Generating (max %d tokens)...", t1 - t0, MAX_NEW_TOKENS)

            raw = self._model.answer_question(
                enc_image,
                PROMPT,
                self._tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            t2 = time.monotonic()
            logger.info("Generated (%.2fs). Raw: %s", t2 - t1, raw[:300])

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Robust JSON extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Extract the semantic payload, tolerating fences, prose and truncation.

        A small model will occasionally run out of tokens mid-object. Losing the
        whole response over a missing brace would be worse than returning the
        fields that did arrive, so truncated output is repaired rather than
        rejected — downstream every field is already optional.
        """
        raw = (raw or "").strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = _loads_or_none(raw)

        if data is None:
            start = raw.find("{")
            if start == -1:
                raise ValueError(f"No JSON object found in model output: {raw[:300]}")

            end = raw.rfind("}")
            if end > start:
                data = _loads_or_none(raw[start:end + 1])
            if data is None:
                data = _repair_truncated(raw[start:])

        if not isinstance(data, dict):
            raise ValueError(f"Model output was not a JSON object: {raw[:300]}")

        return _map_keys(data)


def _loads_or_none(text: str) -> object | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _repair_truncated(fragment: str) -> dict | None:
    """Close an unterminated object and retry, dropping the partial tail."""
    for cut in range(len(fragment), 0, -1):
        if fragment[cut - 1] not in ",{[ \n\t":
            continue
        candidate = fragment[:cut].rstrip().rstrip(",")
        for suffix in ("}", "]}", '"}', '"]}'):
            parsed = _loads_or_none(candidate + suffix)
            if isinstance(parsed, dict):
                return parsed
    return None


def _map_keys(data: dict) -> dict:
    """Rename short model keys to pipeline fields and coerce list/scalar shape."""
    out: dict[str, object] = {}
    for short, field in KEY_MAP.items():
        if short not in data:
            continue
        value = data[short]

        if field in LIST_FIELDS:
            if isinstance(value, str):
                # Models sometimes emit "a, b and c" instead of a list.
                parts = [p.strip() for p in value.replace(" and ", ",").split(",")]
                value = [p for p in parts if p]
            elif not isinstance(value, list):
                value = [value]
            out[field] = [str(v).strip() for v in value if str(v).strip()]
        else:
            if isinstance(value, list):
                value = value[0] if value else ""
            out[field] = str(value).strip()

    return out


vision_service = VisionService()
