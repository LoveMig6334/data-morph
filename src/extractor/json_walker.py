"""Pure recursive walker for JSON values.

Given a parsed JSON node (the output of json.loads), walks the structure
and returns a list of PathStats — one entry per unique path observed.

The walker is pure: no I/O, no warnings. The extractor consumes the
PathStats list and derives warnings as a second pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PathStats:
    """Aggregated facts about a single JSON path.

    Path notation uses dotted keys with ``[]`` marking each array hop.
    Examples: ``[].id`` (root-array element field), ``users[].name``
    (key inside an object root), ``[].orders[].price`` (nested array).

    Attributes:
        path: Path string.
        dtypes_seen: Set of dtype names observed at this path (e.g.
            ``{"integer"}`` or ``{"integer", "string"}`` for mixed).
            ``null`` is recorded separately and does NOT promote to mixed.
        occurrence_count: Number of times this path appeared.
        denominator: Number of times this path's parent existed
            (used for presence = occurrence_count / denominator).
        sample_values: First-seen-wins sample of values at this path,
            capped at sample_values_per_path. Empty for container paths.
        max_depth_seen: Depth at which this path lives. Root = 0.
        array_lengths_seen: Lengths of arrays observed at this path.
            Populated only when path ends in ``[]`` (i.e. the path itself
            refers to an array). Empty tuple otherwise.
        max_length: Max string length observed (string paths only).
        min_value: Min numeric value observed (integer/float paths only).
        max_value: Max numeric value observed (integer/float paths only).
        unique_count: Number of distinct values within sample_values
            (bounded by sample_values_per_path).
    """

    path: str
    dtypes_seen: frozenset[str]
    occurrence_count: int
    denominator: int
    sample_values: tuple[Any, ...]
    max_depth_seen: int
    array_lengths_seen: tuple[int, ...]
    max_length: int | None
    min_value: float | int | None
    max_value: float | int | None
    unique_count: int


@dataclass
class _Acc:
    """Mutable accumulator used during a walk. Converted to PathStats at the end."""

    path: str
    dtypes_seen: set[str] = field(default_factory=set)
    occurrence_count: int = 0
    denominator: int = 0
    sample_values: list[Any] = field(default_factory=list)
    max_depth_seen: int = 0
    array_lengths_seen: list[int] = field(default_factory=list)
    max_length: int | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None


def _dtype_of(value: Any) -> str:
    """Map a JSON-decoded Python value to its dtype name (see spec §5.5)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _record_leaf(acc: _Acc, value: Any, depth: int, cap: int) -> None:
    """Update an accumulator with a single observed leaf value."""
    acc.occurrence_count += 1
    acc.max_depth_seen = max(acc.max_depth_seen, depth)
    acc.dtypes_seen.add(_dtype_of(value))
    if value not in acc.sample_values and len(acc.sample_values) < cap:
        acc.sample_values.append(value)


def _finalize(acc: _Acc) -> PathStats:
    """Freeze a mutable accumulator into an immutable PathStats."""
    return PathStats(
        path=acc.path,
        dtypes_seen=frozenset(acc.dtypes_seen),
        occurrence_count=acc.occurrence_count,
        denominator=acc.denominator,
        sample_values=tuple(acc.sample_values),
        max_depth_seen=acc.max_depth_seen,
        array_lengths_seen=tuple(acc.array_lengths_seen),
        max_length=acc.max_length,
        min_value=acc.min_value,
        max_value=acc.max_value,
        unique_count=len(set(acc.sample_values)),
    )


def walk(root: Any, sample_values_per_path: int) -> list[PathStats]:
    """Walk a parsed JSON value and return per-path statistics.

    For scalar / heterogeneous roots, returns an empty list — the caller
    (JSONExtractor) is responsible for short-circuiting with
    ROOT_SHAPE_UNSUPPORTED.

    Args:
        root: A value as produced by json.loads (dict, list, str, int,
            float, bool, or None).
        sample_values_per_path: Cap on how many distinct values to keep
            per path. Must be > 0.

    Returns:
        Ordered list of PathStats (one per unique path observed).
    """
    if not isinstance(root, (dict, list)):
        return []

    accs: dict[str, _Acc] = {}

    def get(path: str) -> _Acc:
        if path not in accs:
            accs[path] = _Acc(path=path)
        return accs[path]

    def visit_value(value: Any, path: str, depth: int, denom_increment: int) -> None:
        acc = get(path)
        acc.denominator += denom_increment

        # Only record scalar leaves in sample_values; containers update
        # occurrence_count + dtype + max_depth manually so their dtype
        # ("object"/"array") is recorded without polluting sample_values.
        if not isinstance(value, (dict, list)):
            _record_leaf(acc, value, depth=depth, cap=sample_values_per_path)
        else:
            acc.occurrence_count += 1
            acc.max_depth_seen = max(acc.max_depth_seen, depth)
            acc.dtypes_seen.add(_dtype_of(value))

        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                visit_value(child, child_path, depth + 1, denom_increment=1)
        elif isinstance(value, list):
            acc.array_lengths_seen.append(len(value))
            child_path = f"{path}[]"
            for element in value:
                if isinstance(element, dict):
                    for key, sub in element.items():
                        leaf_path = f"{child_path}.{key}"
                        visit_value(sub, leaf_path, depth + 2, denom_increment=1)
                else:
                    visit_value(element, child_path, depth + 1, denom_increment=1)

    if isinstance(root, list):
        for element in root:
            if isinstance(element, dict):
                for key, value in element.items():
                    visit_value(value, f"[].{key}", depth=1, denom_increment=1)
            # Non-dict elements at array root contribute to HETEROGENEOUS_ARRAY
            # detection in the extractor, not here.
    elif isinstance(root, dict):
        for key, value in root.items():
            visit_value(value, key, depth=1, denom_increment=1)

    return [_finalize(acc) for acc in accs.values()]
