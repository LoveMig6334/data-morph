"""Stage 4 — run an Opus-authored conversion script in a subprocess.

Trusted-but-buggy execution: a timeout and (POSIX) CPU limit guard against
runaway scripts. We intentionally do not cap virtual memory (RLIMIT_AS),
because pandas/numpy reserve large virtual address space and would crash.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_SEC = 15.0
DEFAULT_CPU_SECONDS = 15


@dataclass(frozen=True)
class SandboxResult:
    output_text: str
    returncode: int
    stderr: str
    elapsed_sec: float
    error_kind: str  # "ok" | "syntax" | "runtime" | "timeout" | "empty_output"

    @property
    def ok(self) -> bool:
        return self.error_kind == "ok"


def _posix_limits(cpu_seconds: int):
    def _apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    return _apply


def run_script(
    script: str,
    input_path: Path,
    *,
    output_suffix: str,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
) -> SandboxResult:
    """Write `script` to a temp dir, run it on `input_path`, return the output."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        script_path = tmpdir / "convert.py"
        out_path = tmpdir / f"output{output_suffix}"
        script_path.write_text(script, encoding="utf-8")

        preexec = _posix_limits(cpu_seconds) if os.name == "posix" else None
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path), str(input_path), str(out_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=str(tmpdir),
                preexec_fn=preexec,  # type: ignore[arg-type]
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                output_text="",
                returncode=-1,
                stderr=f"Timed out after {timeout_sec}s",
                elapsed_sec=time.perf_counter() - start,
                error_kind="timeout",
            )
        elapsed = time.perf_counter() - start

        if proc.returncode != 0:
            kind = "syntax" if "SyntaxError" in proc.stderr else "runtime"
            return SandboxResult("", proc.returncode, proc.stderr, elapsed, kind)

        if not out_path.exists():
            return SandboxResult("", 0, proc.stderr, elapsed, "empty_output")
        output_text = out_path.read_text(encoding="utf-8")
        if not output_text.strip():
            return SandboxResult(output_text, 0, proc.stderr, elapsed, "empty_output")

        return SandboxResult(output_text, 0, proc.stderr, elapsed, "ok")
