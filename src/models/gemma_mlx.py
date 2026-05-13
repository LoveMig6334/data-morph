"""Gemma 4 E2B IT (bf16) inference via mlx-vlm.

Module-level singleton: load() is called once on the first generate() call;
subsequent calls reuse the same model + processor. mlx-vlm imports are lazy
(inside generate) so this module is importable on non-MLX machines without
crashing — it only fails when generate() is actually invoked.

Why mlx-vlm (not mlx-lm): Gemma 4's E2B variant is multimodal ("Any-to-Any"),
shipped with a `language_model.*` prefix on its safetensors. mlx-lm is
text-only and cannot strip that prefix, so it errors at load. mlx-vlm is
the multimodal counterpart and knows how to dispatch on the wrapper; for
our text-only inference path we pass `num_images=0` to `apply_chat_template`
and no image/audio/video args to `generate`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

MODEL_ID = "mlx-community/gemma-4-e2b-it-bf16"

_state: dict = {"model": None, "processor": None, "load_sec": None}


@dataclass
class GenerationResult:
    text: str
    n_prompt_tokens: int
    n_generated_tokens: int
    elapsed_sec: float
    tokens_per_sec: float
    model_id: str
    truncated: bool


def _ensure_loaded() -> None:
    if _state["model"] is not None:
        return
    from mlx_vlm import load  # lazy import

    t0 = time.time()
    model, processor = load(MODEL_ID)
    _state["model"] = model
    _state["processor"] = processor
    _state["load_sec"] = round(time.time() - t0, 2)
    print(f"[gemma_mlx] loaded {MODEL_ID} in {_state['load_sec']}s")


def generate(messages: list[dict], max_tokens: int = 4096) -> GenerationResult:
    """Run greedy generation on the given chat messages (text-only)."""
    _ensure_loaded()
    from mlx_vlm import generate as vlm_generate  # lazy
    from mlx_vlm.prompt_utils import apply_chat_template  # lazy

    model = _state["model"]
    processor = _state["processor"]

    prompt = apply_chat_template(
        processor, model.config, messages, num_images=0
    )

    t0 = time.time()
    result = vlm_generate(
        model=model,
        processor=processor,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    elapsed = time.time() - t0

    n_prompt = int(getattr(result, "prompt_tokens", 0) or 0)
    n_gen = int(getattr(result, "generation_tokens", 0) or 0)
    if elapsed > 0:
        tps = float(getattr(result, "generation_tps", n_gen / elapsed) or 0.0)
    else:
        tps = 0.0
    truncated = n_gen >= max_tokens

    return GenerationResult(
        text=result.text,
        n_prompt_tokens=n_prompt,
        n_generated_tokens=n_gen,
        elapsed_sec=round(elapsed, 2),
        tokens_per_sec=round(tps, 2),
        model_id=MODEL_ID,
        truncated=truncated,
    )
