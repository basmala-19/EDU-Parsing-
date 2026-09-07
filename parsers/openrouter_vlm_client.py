"""
OpenRouter cloud VLM client — optional cloud-assist backend for the Qwen
refinement layer in ``educational/llm_parser.py``.

Why this exists
----------------
``QwenVLRefiner`` and ``QwenPageParser`` already accept a pluggable
``model_client`` callable with the signature ``(image: PIL.Image, prompt:
str) -> str``. By default they load a *local* Qwen2.5-VL model, which needs
a GPU, several GB of VRAM, and a model download on first use. This module
is a drop-in alternative that sends the same image + prompt to a hosted
vision-language model on OpenRouter instead, so the exact same refinement
logic (needs_refinement / _apply / prompts) works with no local GPU at all:

    from parsers.openrouter_vlm_client import OpenRouterVLMClient
    from educational.llm_parser import QwenVLRefiner

    client = OpenRouterVLMClient()          # reads OPENROUTER_API_KEY from env
    report = QwenVLRefiner(model_client=client).refine(edoc, pdf_path)

Model choice
------------
Default model: ``google/gemini-3-flash-preview`` (Gemini 3 Flash Preview).

Deliberately *not* another Qwen model — the point of the cloud backend is
to bring in a different model family so the two disagree on different
things instead of sharing the same blind spots (this project's local
refiner already runs Qwen2.5-VL-3B). Flash was picked over Gemini 3 Pro
because ``QwenVLRefiner.needs_refinement()`` only flags a small slice of
elements per document (low-confidence OCR, empty tables, placeholder image
captions) — not every page — so this call runs many times per document but
on small, already-isolated crops. Flash scores close to Pro on
multilingual/table extraction while costing ~4x less on input and ~4x less
on output (see pricing below), which matters a lot at that call volume.
Pro is kept available as an escalation option, not the default.

Pricing (OpenRouter, checked 2026-09-01 — always re-check before relying on
this for budgeting, these preview models get repriced):
    gemini-3-flash-preview : $0.50 / 1M input tok,  $3.00 / 1M output tok
    gemini-3-pro-preview   : $2.00 / 1M input tok,  $12.00 / 1M output tok
    qwen3-vl-235b-a22b-instruct : $0.20 / 1M input tok, $0.88 / 1M output tok
        (cheapest of the three, but it's the same model family as the local
        Qwen2.5-VL refiner, so it's a weaker second opinion than either
        Gemini option — kept as a low-cost fallback, not the default.)

Set a different model at any time via ``.env``:

    OPENROUTER_MODEL=google/gemini-3-pro-preview      # escalate for hard tables
    OPENROUTER_MODEL=qwen/qwen3-vl-235b-a22b-instruct  # cheapest option

Any other vision-capable chat model on OpenRouter (GPT-5 class, etc.) also
works — this client only assumes an OpenAI-style ``/chat/completions``
endpoint with an ``image_url`` content block, which OpenRouter guarantees
for every vision model it lists.

Cost/scope note
----------------
Nothing in this project calls OpenRouter automatically. It only runs when
you explicitly pass ``refinement_backend="openrouter"`` to ``run_pipeline``
(or tick the matching option in ``app.py``'s sidebar), and even then only
for the small number of elements ``QwenVLRefiner.needs_refinement()``
flags — low-confidence OCR, empty tables, placeholder image captions — not
for every page of every book.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path

import requests
from PIL import Image

# Load .env ourselves instead of relying on some other module (previously only
# parsers/llama_parser.py did this) having been imported first. Without this,
# constructing OpenRouterVLMClient() directly/standalone -- exactly the usage
# shown in the module docstring above -- silently sees an empty environment:
# OPENROUTER_API_KEY looks "missing" even though it's correctly set in .env,
# and OPENROUTER_MODEL is silently ignored in favor of DEFAULT_MODEL below.
# load_dotenv() never overrides variables already set in the real environment,
# so this is safe to call even if the process set OPENROUTER_API_KEY itself.
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"

# Strong multilingual (Arabic + English) document/OCR/table model at a
# fraction of Pro's cost. See the module docstring above for the reasoning,
# pricing, and how to escalate to Pro or drop to Qwen for cheaper runs.
DEFAULT_MODEL = "google/gemini-3-flash-preview"


class OpenRouterVLMClient:
    """Callable ``(image, prompt) -> str`` backed by an OpenRouter vision model.

    Pass an instance of this class as ``model_client=`` to ``QwenVLRefiner``
    or ``QwenPageParser`` (see ``educational/llm_parser.py``) to run their
    existing refinement/parsing logic against a hosted model instead of a
    local one.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_tokens: int = 1500,
        site_url: str | None = None,
        site_name: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to edu2/.env (copy "
                ".env.example to .env first if you haven't) or pass api_key= "
                "explicitly. If you already added it to .env and still see this: "
                "the file must be named exactly '.env' (not '.env.example' or "
                "'.env.txt') and live at edu22/edu2/.env. "
                "Get a key at https://openrouter.ai/keys"
            )
        env_model = os.environ.get("OPENROUTER_MODEL")
        self.model = model or env_model or DEFAULT_MODEL
        if model is None and env_model is None:
            logger.warning(
                "OPENROUTER_MODEL not set (env not loaded, or var missing) — "
                "falling back to DEFAULT_MODEL=%s instead of whatever you intended "
                "(e.g. a Gemma/Qwen model). Set OPENROUTER_MODEL in edu2/.env "
                "or pass model= explicitly.",
                DEFAULT_MODEL,
            )
        else:
            logger.info("OpenRouterVLMClient using model=%s", self.model)
        self.timeout = timeout
        self.max_tokens = max_tokens
        # Optional, but OpenRouter uses these for its public leaderboards /
        # rate-limit attribution. Harmless to omit.
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL", "")
        self.site_name = site_name or os.environ.get(
            "OPENROUTER_SITE_NAME", "Educational RAG Parsing Pipeline"
        )

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        # Preprocessing: limit max dimension to 1600px to avoid timeouts on VLM providers
        max_dim = 1600
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=82, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers

    def __call__(self, image: Image.Image, prompt: str) -> str:
        """Send one image + prompt to OpenRouter and return the text response.

        Raises RuntimeError with the response body on any non-200 status or
        an unexpected payload shape, so failures surface clearly instead of
        silently returning an empty string (the caller — QwenVLRefiner —
        already treats any exception here as "reject this refinement").
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        import time
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    OPENROUTER_CHAT_COMPLETIONS_URL,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"] or ""
                    raise RuntimeError(f"Unexpected OpenRouter response shape: {data}")

                # If upstream provider timeout (504) or bad gateway (502/520) / rate limit (429), retry
                if response.status_code in (429, 502, 503, 504, 520) and attempt < max_retries:
                    time.sleep(attempt * 2)
                    continue

                raise RuntimeError(
                    f"OpenRouter request failed ({response.status_code}): {response.text[:500]}"
                )
            except (requests.exceptions.RequestException, RuntimeError) as exc:
                last_error = exc
                err_text = str(exc)
                if any(code in err_text for code in ("504", "520", "502", "10054", "Connection aborted", "Timeout")) and attempt < max_retries:
                    time.sleep(attempt * 2)
                    continue
                raise last_error

        raise last_error or RuntimeError("OpenRouter failed after retries")
