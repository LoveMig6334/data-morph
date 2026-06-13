from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

_LOCAL_PATH = Path(__file__).resolve().parents[2] / "models" / "gemma-4-e2b-it-bf16"
# Base model dir; override via GEMMA_MLX_MODEL to point at e.g. a quantized build.
MODEL_ID = os.environ.get("GEMMA_MLX_MODEL", str(_LOCAL_PATH))

# When the model is a stripped text-only build (gemma4_text), load it through
# mlx_lm instead of mlx_vlm — there is no vision/audio tower to wrap. Set
# GEMMA_TEXT_ONLY=1 to select this path (used by the W7 text-only student).
TEXT_ONLY = os.environ.get("GEMMA_TEXT_ONLY", "").lower() in ("1", "true", "yes")

# Model selection lives in _state so it can be changed at runtime (via use_model)
# without re-importing the module. The env vars seed the defaults, so existing
# scripts/notebooks that set GEMMA_MLX_MODEL / GEMMA_TEXT_ONLY keep working.
_state: dict = {
    "model": None,
    "processor": None,
    "load_sec": None,
    "adapter": None,
    "model_id": MODEL_ID,
    "text_only": TEXT_ONLY,
}


def use_adapter(adapter_path: str | None) -> None:
    """Select a LoRA adapter directory for inference (None = base model).

    The directory must contain ``adapters.safetensors`` + ``adapter_config.json``.
    Changing the selection forces the model to reload on the next ``generate`` call.
    """
    if adapter_path != _state["adapter"]:
        _state["adapter"] = adapter_path
        _state["model"] = None  # force reload with the new adapter (or base)


def use_model(model_id: str, *, text_only: bool = True) -> None:
    """Select which model to load (local dir path or HF id).

    Mirrors ``use_adapter``: changing the selection forces a reload on the next
    ``generate`` call. ``text_only=True`` loads via mlx_lm (the stripped W7
    student); ``False`` uses mlx_vlm (the multimodal base).
    """
    if model_id != _state["model_id"] or text_only != _state["text_only"]:
        _state["model_id"] = model_id
        _state["text_only"] = text_only
        _state["model"] = None  # force reload with the new model


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
    t0 = time.time()
    model_id = _state["model_id"]
    text_only = _state["text_only"]
    if text_only:
        import mlx_lm  # lazy import

        loaded = mlx_lm.load(model_id, adapter_path=_state["adapter"])
        model, processor = loaded[0], loaded[1]
    else:
        from mlx_vlm import load  # lazy import

        model, processor = load(model_id, adapter_path=_state["adapter"])
    _state["model"] = model
    _state["processor"] = processor
    _state["load_sec"] = round(time.time() - t0, 2)
    tag = f" + adapter {_state['adapter']}" if _state["adapter"] else ""
    backend = "mlx_lm/text" if text_only else "mlx_vlm"
    print(f"[gemma_mlx] loaded {model_id}{tag} via {backend} in {_state['load_sec']}s")


def _generate_text_only(messages: list[dict], max_tokens: int) -> GenerationResult:
    """Greedy generation via mlx_lm on the stripped text-only student."""
    import mlx_lm  # lazy
    from mlx_lm.sample_utils import make_sampler  # lazy

    model = _state["model"]
    tok = _state["processor"]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    t0 = time.time()
    text = mlx_lm.generate(
        model,
        tok,
        str(prompt),
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0),  # greedy, matches the temperature=0.0 VLM path
        verbose=False,
    )
    elapsed = time.time() - t0

    n_prompt = len(tok.encode(str(prompt)))
    n_gen = len(tok.encode(text)) if text else 0
    tps = round(n_gen / elapsed, 2) if elapsed > 0 else 0.0
    return GenerationResult(
        text=text,
        n_prompt_tokens=n_prompt,
        n_generated_tokens=n_gen,
        elapsed_sec=round(elapsed, 2),
        tokens_per_sec=tps,
        model_id=_state["model_id"],
        truncated=n_gen >= max_tokens,
    )


def generate(messages: list[dict], max_tokens: int = 4096) -> GenerationResult:
    """Run greedy generation on the given chat messages (text-only)."""
    _ensure_loaded()
    if _state["text_only"]:
        return _generate_text_only(messages, max_tokens)

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
        model_id=_state["model_id"],
        truncated=truncated,
    )
