"""Pure recursive walker for JSON values.

Given a parsed JSON node (the output of json.loads), walks the structure
and returns a list of PathStats — one entry per unique path observed.

The walker is pure: no I/O, no warnings. The extractor consumes the
PathStats list and derives warnings as a second pass.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    # Implementation in subsequent tasks builds out array-root, object-root,
    # nested cases, and container path emission.
    return []
