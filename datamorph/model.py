"""Resolve which local MLX model the conversion pipeline should load.

For now the model must be present locally (point ``$GEMMA_MLX_MODEL`` at a copy
of the final student, or pass ``model=`` to ``convert_file``). Automatic download
from the Hugging Face Hub is a planned follow-up — once the model is published the
default will fall back to a Hub id instead of raising.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "GEMMA_MLX_MODEL"
DEFAULT_MODEL_NAME = "gemma4-e2b-textonly-iter400-q8-vocab16k"
# datamorph/model.py -> repo root is one parent up; weights live in models/.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / DEFAULT_MODEL_NAME


def resolve_model(model: str | os.PathLike[str] | None = None) -> str:
    """Return the path to the MLX model to load.

    Resolution order: explicit ``model`` argument → ``$GEMMA_MLX_MODEL`` → the
    default local final-model directory. Raises ``FileNotFoundError`` if the
    resolved local path does not exist.
    """
    candidate = model or os.environ.get(ENV_VAR) or DEFAULT_MODEL_DIR
    path = Path(candidate).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Point ${ENV_VAR} at a local copy of the "
            f"final student ({DEFAULT_MODEL_NAME}), or pass model=... to "
            "convert_file(). Hugging Face Hub download is not yet available."
        )
    return str(path)
