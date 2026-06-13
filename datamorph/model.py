"""Resolve which MLX model the conversion pipeline should load.

Resolution order (first hit wins):

1. an explicit ``model`` argument, or ``$GEMMA_MLX_MODEL`` — a local directory
   path, or a Hugging Face repo id;
2. the default local model directory, if present (skips a download for
   developers who already have the weights);
3. the published model on the Hugging Face Hub, downloaded + cached.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV_VAR = "GEMMA_MLX_MODEL"
DEFAULT_MODEL_NAME = "gemma4-e2b-textonly-iter400-q8-vocab16k"
DEFAULT_HF_REPO = "Bunnana/data-morph-gemma-2b"
# datamorph/model.py -> repo root is one parent up; weights live in models/.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / DEFAULT_MODEL_NAME

# A Hugging Face repo id is "<owner>/<name>" — exactly one slash, no leading
# slash, and only id-safe characters. Anything else is treated as a local path.
_REPO_ID_RE = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+\Z")


def _looks_like_repo_id(value: str) -> bool:
    return bool(_REPO_ID_RE.fullmatch(value))


def _download(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "huggingface_hub is required to download the model. "
            "Install it with `pip install huggingface_hub`."
        ) from exc
    return snapshot_download(repo_id)


def resolve_model(model: str | os.PathLike[str] | None = None) -> str:
    """Return a local path to the MLX model to load (downloading it if needed)."""
    candidate = model or os.environ.get(ENV_VAR)
    if candidate:
        candidate = str(candidate)
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
        if _looks_like_repo_id(candidate):
            return _download(candidate)
        raise FileNotFoundError(
            f"Model path not found: {path}. Pass a local directory, a Hugging Face "
            f"repo id, or set ${ENV_VAR}."
        )

    if DEFAULT_MODEL_DIR.exists():
        return str(DEFAULT_MODEL_DIR)

    # Nothing local — fall back to the published model on the Hub.
    return _download(DEFAULT_HF_REPO)
