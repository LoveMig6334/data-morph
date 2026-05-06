"""Four evaluation metrics for file-format conversion outputs.

All functions are pure: they take strings (not paths), return floats in [0.0, 1.0],
and do not touch the filesystem or network. This makes them unit-testable.

Metrics (per the AI Builders 2026 project plan):
    1. format_validity      — output is a valid file in the target format
    2. schema_compliance    — output matches expected structure (keys / columns)
    3. loadability          — pandas can consume the output
    4. content_accuracy     — field-level / substring match against expected
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 1. Format validity
# ---------------------------------------------------------------------------


def format_validity(output: str, output_format: str) -> float:
    """Return 1.0 if output parses as the target format, else 0.0."""
    fmt = output_format.lower()
    if fmt == "json":
        try:
            json.loads(output)
            return 1.0
        except (json.JSONDecodeError, ValueError):
            return 0.0
    if fmt == "csv":
        try:
            reader = csv.reader(io.StringIO(output))
            rows = list(reader)
            if not rows:
                return 0.0
            width = len(rows[0])
            if width == 0:
                return 0.0
            # every row must have the same column count
            if not all(len(r) == width for r in rows):
                return 0.0
            return 1.0
        except csv.Error:
            return 0.0
    if fmt == "txt":
        return 1.0 if output.strip() else 0.0
    raise ValueError(f"Unknown output_format: {output_format!r}")


# ---------------------------------------------------------------------------
# 2. Schema compliance
# ---------------------------------------------------------------------------


def _json_key_skeleton(obj: Any) -> Any:
    """Recursively reduce a JSON value to its structural skeleton.

    Dicts -> sorted tuple of (key, child_skeleton).
    Lists -> ('list', child_skeleton_of_first) so we check the per-element shape
             rather than length (caller decides whether length matters).
    Scalars -> the type name.
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, _json_key_skeleton(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        if not obj:
            return ("list", "empty")
        # Use first element's skeleton as representative; a well-formed output
        # should have homogeneous elements in each array position.
        return ("list", _json_key_skeleton(obj[0]))
    return type(obj).__name__


def schema_compliance(actual: str, expected: str, output_format: str) -> float:
    """Return 1.0 if actual's structural skeleton matches expected's, else 0.0."""
    fmt = output_format.lower()
    if fmt == "json":
        try:
            a = json.loads(actual)
            e = json.loads(expected)
        except (json.JSONDecodeError, ValueError):
            return 0.0
        return 1.0 if _json_key_skeleton(a) == _json_key_skeleton(e) else 0.0
    if fmt == "csv":
        try:
            a_rows = list(csv.reader(io.StringIO(actual)))
            e_rows = list(csv.reader(io.StringIO(expected)))
        except csv.Error:
            return 0.0
        if not a_rows or not e_rows:
            return 0.0
        # header match (case-insensitive, trimmed)
        a_hdr = [c.strip().lower() for c in a_rows[0]]
        e_hdr = [c.strip().lower() for c in e_rows[0]]
        return 1.0 if a_hdr == e_hdr else 0.0
    if fmt == "txt":
        # No meaningful structural check for freeform TXT.
        return 1.0
    raise ValueError(f"Unknown output_format: {output_format!r}")


# ---------------------------------------------------------------------------
# 3. Loadability
# ---------------------------------------------------------------------------


def loadability(output: str, output_format: str) -> float:
    """Return 1.0 if pandas can load the output without error, else 0.0."""
    fmt = output_format.lower()
    if fmt == "json":
        try:
            import pandas as pd

            data = json.loads(output)
            # pd.json_normalize handles both lists-of-objects and nested dicts.
            if isinstance(data, list):
                pd.json_normalize(data)
            elif isinstance(data, dict):
                # normalize the first list-valued field if present, else wrap.
                list_fields = [v for v in data.values() if isinstance(v, list)]
                if list_fields:
                    pd.json_normalize(list_fields[0])
                else:
                    pd.json_normalize([data])
            else:
                return 0.0
            return 1.0
        except Exception:
            return 0.0
    if fmt == "csv":
        try:
            import pandas as pd

            df = pd.read_csv(io.StringIO(output))
            return 1.0 if len(df.columns) > 0 else 0.0
        except Exception:
            return 0.0
    if fmt == "txt":
        return 1.0 if output.strip() else 0.0
    raise ValueError(f"Unknown output_format: {output_format!r}")


# ---------------------------------------------------------------------------
# 4. Content accuracy
# ---------------------------------------------------------------------------


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two scalar values with light coercion.

    - Numeric strings compare equal to numbers: "9.99" == 9.99.
    - None == "". ("null" is handled by JSON already being None.)
    - Strings compare case-sensitive after .strip().
    """
    if a == b:
        return True
    # Both numeric (possibly as strings)?
    try:
        fa, fb = float(a), float(b)
        if fa == fb:
            return True
    except (TypeError, ValueError):
        pass
    # Both empty-ish?
    if (a is None or a == "") and (b is None or b == ""):
        return True
    # String comparison with whitespace strip
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return False


def _walk_json_leaves(obj: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    """Yield (key_path, leaf_value) pairs from a JSON-decoded object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            yield from _walk_json_leaves(v, new_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            yield from _walk_json_leaves(v, new_path)
    else:
        yield path, obj


def _json_content_accuracy(actual_text: str, expected_text: str) -> float:
    """Fraction of expected leaf paths that match actual."""
    try:
        actual = json.loads(actual_text)
        expected = json.loads(expected_text)
    except (json.JSONDecodeError, ValueError):
        return 0.0
    actual_map = dict(_walk_json_leaves(actual))
    expected_map = dict(_walk_json_leaves(expected))
    if not expected_map:
        return 0.0
    matches = sum(
        1
        for path, ev in expected_map.items()
        if path in actual_map and _values_equal(actual_map[path], ev)
    )
    return matches / len(expected_map)


def _csv_content_accuracy(actual_text: str, expected_text: str) -> float:
    """Fraction of expected cells that match actual (by header-aware row alignment).

    Rows are aligned positionally; cells are compared by shared column name.
    If the header differs, score is 0.0 (that's a schema-compliance issue).
    """
    try:
        a_rows = list(csv.reader(io.StringIO(actual_text)))
        e_rows = list(csv.reader(io.StringIO(expected_text)))
    except csv.Error:
        return 0.0
    if len(a_rows) < 1 or len(e_rows) < 1:
        return 0.0
    a_hdr = [c.strip() for c in a_rows[0]]
    e_hdr = [c.strip() for c in e_rows[0]]
    if [h.lower() for h in a_hdr] != [h.lower() for h in e_hdr]:
        return 0.0
    a_data, e_data = a_rows[1:], e_rows[1:]
    total = len(e_data) * len(e_hdr)
    if total == 0:
        return 0.0
    matches = 0
    for i, e_row in enumerate(e_data):
        a_row = a_data[i] if i < len(a_data) else [""] * len(e_hdr)
        for j, e_cell in enumerate(e_row):
            a_cell = a_row[j] if j < len(a_row) else ""
            if _values_equal(a_cell.strip(), e_cell.strip()):
                matches += 1
    return matches / total


def _txt_content_accuracy(actual_text: str, required_substrings: list[str]) -> float:
    """Fraction of required substrings present in actual (case-insensitive)."""
    if not required_substrings:
        return 0.0
    hay = actual_text.lower()
    hits = sum(1 for s in required_substrings if s.lower() in hay)
    return hits / len(required_substrings)


def content_accuracy(
    actual: str,
    expected: str,
    output_format: str,
    required_substrings: list[str] | None = None,
) -> float:
    """Dispatch to the format-appropriate content-accuracy routine."""
    fmt = output_format.lower()
    if fmt == "json":
        return _json_content_accuracy(actual, expected)
    if fmt == "csv":
        return _csv_content_accuracy(actual, expected)
    if fmt == "txt":
        return _txt_content_accuracy(actual, required_substrings or [])
    raise ValueError(f"Unknown output_format: {output_format!r}")


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------


def score_all(
    actual: str,
    expected: str,
    output_format: str,
    required_substrings: list[str] | None = None,
) -> dict[str, float]:
    """Run all four metrics and return a dict of scores."""
    return {
        "format_validity": format_validity(actual, output_format),
        "schema_compliance": schema_compliance(actual, expected, output_format),
        "loadability": loadability(actual, output_format),
        "content_accuracy": content_accuracy(
            actual, expected, output_format, required_substrings
        ),
    }
