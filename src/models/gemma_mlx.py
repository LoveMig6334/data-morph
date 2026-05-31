from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

_LOCAL_PATH = Path(__file__).resolve().parents[2] / "models" / "gemma-4-e2b-it-bf16"
# Base model dir; override via GEMMA_MLX_MODEL to point at e.g. a quantized build.
MODEL_ID = os.environ.get("GEMMA_MLX_MODEL", str(_LOCAL_PATH))

_state: dict = {"model": None, "processor": None, "load_sec": None, "adapter": None}


def use_adapter(adapter_path: str | None) -> None:
    """Select a LoRA adapter directory for inference (None = base model).

    The directory must contain ``adapters.safetensors`` + ``adapter_config.json``.
    Changing the selection forces the model to reload on the next ``generate`` call.
    """
    if adapter_path != _state["adapter"]:
        _state["adapter"] = adapter_path
        _state["model"] = None  # force reload with the new adapter (or base)


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
    model, processor = load(MODEL_ID, adapter_path=_state["adapter"])
    _state["model"] = model
    _state["processor"] = processor
    _state["load_sec"] = round(time.time() - t0, 2)
    tag = f" + adapter {_state['adapter']}" if _state["adapter"] else ""
    print(f"[gemma_mlx] loaded {MODEL_ID}{tag} in {_state['load_sec']}s")


def generate(messages: list[dict], max_tokens: int = 4096) -> GenerationResult:
    """Run greedy generation on the given chat messages (text-only)."""
    _ensure_loaded()
    from mlx_vlm import generate as vlm_generate  # lazy
    from mlx_vlm.prompt_utils import apply_chat_template  # lazy

    model = _state["model"]
    processor = _state["processor"]

    prompt = apply_chat_template(processor, model.config, messages, num_images=0)

    t0 = time.time()
    result = vlm_generate(
        model=model,
        processor=processor,
        prompt=str(prompt),
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
