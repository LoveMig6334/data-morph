"""Gemma 2 2B IT (bf16) inference via MLX.

Module-level singleton: load() is called once on the first generate() call;
subsequent calls reuse the same model + tokenizer. mlx_lm imports are lazy
(inside generate) so this module is importable on non-MLX machines without
crashing — it only fails when generate() is actually invoked.

Why Gemma 2 (not Gemma 4): the Gemma 4 family has no text-only 2B variant —
E2B and E4B are multimodal ("Any-to-Any") and mlx-lm can't load them because
their safetensors are prefixed `language_model.*` (multimodal wrapper) which
mlx-lm doesn't strip. Gemma 2 2B IT is text-only, true 2B, and matches the
project plan's stated student-model target.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

MODEL_ID = "mlx-community/gemma-2-2b-it"

_state: dict = {"model": None, "tokenizer": None, "load_sec": None}


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
    from mlx_lm import load  # lazy import

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["load_sec"] = round(time.time() - t0, 2)
    print(f"[gemma_mlx] loaded {MODEL_ID} in {_state['load_sec']}s")


def generate(messages: list[dict], max_tokens: int = 4096) -> GenerationResult:
    """Run greedy generation on the given chat messages."""
    _ensure_loaded()
    from mlx_lm import generate as mlx_generate  # lazy
    from mlx_lm.sample_utils import make_sampler  # lazy

    tokenizer = _state["tokenizer"]
    model = _state["model"]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer.encode(prompt)
    n_prompt = len(prompt_ids)

    sampler = make_sampler(temp=0.0)
    t0 = time.time()
    text = mlx_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    elapsed = time.time() - t0

    gen_ids = tokenizer.encode(text, add_special_tokens=False)
    n_gen = len(gen_ids)
    tps = round(n_gen / elapsed, 2) if elapsed > 0 else 0.0
    truncated = n_gen >= max_tokens

    return GenerationResult(
        text=text,
        n_prompt_tokens=n_prompt,
        n_generated_tokens=n_gen,
        elapsed_sec=round(elapsed, 2),
        tokens_per_sec=tps,
        model_id=MODEL_ID,
        truncated=truncated,
    )
