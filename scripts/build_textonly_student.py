"""Build a text-only, adapter-fused student (W7 model-surgery, steps 1+2).

Two lossless operations on the bf16 base:

  1. **Fuse** a LoRA adapter into the language weights. For each adapter target
     the inference math is ``y = Wx + scale * ((xA)B)`` with ``scale = alpha/rank``
     (mlx_vlm ``LoRaLayer``), so the folded weight is ``W += scale * (B.T @ A.T)``
     — identical to mlx_lm's ``LoRALinear.fuse``. Adapter only touches
     ``language_model.*`` modules, so nothing outside the text tower changes.

  2. **Strip** the unused vision + audio towers. The conversion pipeline never
     feeds an image or audio token, so ``vision_tower`` / ``audio_tower`` /
     ``embed_vision`` / ``embed_audio`` are dead weight (~0.95 GB of bf16 that
     quantization never even compresses). Dropping them leaves a pure
     ``gemma4_text`` model that **mlx_lm loads directly** — no VLM wrapper, no
     image/audio code path.

The result is a standalone mlx_lm model dir (config.json + one safetensors shard
+ tokenizer), loadable with ``mlx_lm.load(out_dir)``.

NOTE: in bf16 this stage only removes the ~0.95 GB vision/audio towers (PLE and
the 262 k-row embeddings still ship). The large reductions come later from vocab
pruning (step 3) and quantization (step 4). The point of this stage is to PROVE
accuracy survives the fuse + strip before anything lossy happens.

Usage:
    uv run python scripts/build_textonly_student.py \
        --adapter models/lora_gemma4e2b_scriptgen/0000400_adapters.safetensors \
        --out models/gemma4-e2b-textonly-iter400-bf16
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import cast

import mlx.core as mx


def _load_dict(path: str) -> dict[str, mx.array]:
    """mx.load on a safetensors file always yields a name->array dict."""
    return cast("dict[str, mx.array]", mx.load(path))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "models" / "gemma-4-e2b-it-bf16"

# Tokenizer / template files copied verbatim so the text-only model formats
# prompts identically to the VLM it was distilled from.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "generation_config.json",
]


def load_base_weights(base: Path) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    shards = sorted(glob.glob(str(base / "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {base}")
    for f in shards:
        weights.update(_load_dict(f))
    return weights


def fuse_adapter(
    weights: dict[str, mx.array], adapter_file: Path, rank: int, alpha: float
) -> tuple[int, float]:
    """Fold LoRA A/B into the base weights in place. Returns (n_fused, scale)."""
    scale = alpha / rank
    ad = _load_dict(str(adapter_file))
    targets = sorted({k[:-2] for k in ad if k.endswith(".A")})
    if not targets:
        raise ValueError(f"{adapter_file} has no .A/.B LoRA tensors")
    n = 0
    for t in targets:
        A = ad[t + ".A"]  # (in, r)
        B = ad[t + ".B"]  # (r, out)
        wkey = t + ".weight"
        if wkey not in weights:
            raise KeyError(f"adapter target {t!r} has no base weight {wkey!r}")
        W = weights[wkey]  # (out, in)
        delta = scale * (B.T @ A.T)  # (out, in)
        if delta.shape != W.shape:
            raise ValueError(f"shape mismatch at {t}: W{W.shape} vs delta{delta.shape}")
        weights[wkey] = W + delta.astype(W.dtype)
        n += 1
    return n, scale


def to_text_only(
    weights: dict[str, mx.array], n_layers: int, n_kv_shared: int
) -> tuple[dict[str, mx.array], int, int]:
    """Keep only the language tower; rename language_model.model.X -> model.X.

    Also drops the redundant K/V projections on KV-shared layers. Gemma4 shares
    the KV of the last non-shared layer across the final ``n_kv_shared`` layers,
    so mlx_lm's ``gemma4_text`` never allocates ``k_proj``/``v_proj``/``k_norm``
    for layer indices >= ``n_layers - n_kv_shared``. The HF checkpoint still
    stores them (and mlx_vlm just ignored them via strict=False at load), so we
    drop them here to match the architecture exactly.
    """
    first_shared = n_layers - n_kv_shared if n_kv_shared > 0 else n_layers
    shared_idx = set(range(first_shared, n_layers))
    shared_suffixes = ("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight")

    out: dict[str, mx.array] = {}
    dropped_nontext = 0
    dropped_sharedkv = 0
    for k, v in weights.items():
        if not k.startswith("language_model."):
            dropped_nontext += 1  # vision_tower / audio_tower / embed_vision / embed_audio
            continue
        nk = k.replace("language_model.model.", "model.", 1)
        if nk.startswith("model.layers."):
            li = int(nk.split(".")[2])
            if li in shared_idx and nk.endswith(shared_suffixes):
                dropped_sharedkv += 1
                continue
        out[nk] = v
    return out, dropped_nontext, dropped_sharedkv


def write_config(base: Path, out: Path) -> None:
    """Flatten text_config to a top-level gemma4_text config mlx_lm can load."""
    full = json.loads((base / "config.json").read_text())
    tc = dict(full["text_config"])
    tc["model_type"] = "gemma4_text"  # already is, but be explicit
    tc.setdefault("tie_word_embeddings", full.get("tie_word_embeddings", True))
    (out / "config.json").write_text(json.dumps(tc, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument(
        "--adapter",
        default=str(
            PROJECT_ROOT / "models/lora_gemma4e2b_scriptgen/0000400_adapters.safetensors"
        ),
        help="LoRA adapter .safetensors to fuse (default: iter-400)",
    )
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading base weights from {base} ...")
    weights = load_base_weights(base)
    print(f"      {len(weights)} tensors")

    print(f"[2/4] fusing adapter {Path(args.adapter).name} (rank={args.rank}, alpha={args.alpha}) ...")
    n_fused, scale = fuse_adapter(weights, Path(args.adapter), args.rank, args.alpha)
    print(f"      fused {n_fused} LoRA targets (scale={scale})")

    print("[3/4] stripping vision + audio towers + redundant shared-KV ...")
    full_cfg = json.loads((base / "config.json").read_text())["text_config"]
    text_weights, dropped, dropped_kv = to_text_only(
        weights, full_cfg["num_hidden_layers"], full_cfg["num_kv_shared_layers"]
    )
    n_params = sum(int(v.size) for v in text_weights.values())
    print(f"      kept {len(text_weights)} text tensors; dropped {dropped} non-text + {dropped_kv} shared-KV tensors")
    print(f"      text-only params: {n_params/1e9:.3f} B")

    print(f"[4/4] writing model to {out} ...")
    write_config(base, out)
    for fn in TOKENIZER_FILES:
        src = base / fn
        if src.exists():
            shutil.copy(src, out / fn)
    mx.save_safetensors(str(out / "model.safetensors"), text_weights)

    print(f"\nDone. Text-only fused student -> {out}")
    print("Load with: mlx_lm.load(<out>)  |  eval via GEMMA_TEXT_ONLY=1 GEMMA_MLX_MODEL=<out>")


if __name__ == "__main__":
    main()
