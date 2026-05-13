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
    load_result = load(MODEL_ID)
    model = load_result[0]
    tokenizer = load_result[1]
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
