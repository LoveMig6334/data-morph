"""Generator: flat CSV (user_*/order_*) -> nested JSON (user object + orders array)."""

from __future__ import annotations

import csv
import io
import json
import random

from .base import GeneratedCase, make_faker

_N_USERS = {"simple": 3, "medium": 8, "complex": 20}
_MAX_ORDERS = {"simple": 1, "medium": 3, "complex": 5}
_ITEMS = ["Widget", "Gadget", "Thingamajig", "Doohickey", "Sprocket", "Cog"]


def generate(seed: int, complexity: str) -> GeneratedCase:
    rng = random.Random(seed)
    fake = make_faker(seed)
    n_users = _N_USERS[complexity]
    max_orders = _MAX_ORDERS[complexity]

    rows: list[tuple[str, str, int, str, float]] = []
    records: list[dict] = []
    oid = 1000
    for i in range(n_users):
        name = f"{fake.first_name()}{i}"  # index suffix guarantees uniqueness
        email = f"{name.lower()}@example.com"
        n_orders = 1 if complexity == "simple" else rng.randint(1, max_orders)
        orders = []
        for _ in range(n_orders):
            oid += 1
            item = rng.choice(_ITEMS)
            price = round(rng.uniform(1, 100), 2)
            rows.append((name, email, oid, item, price))
            orders.append({"id": oid, "item": item, "price": price})
        records.append({"user": {"name": name, "email": email}, "orders": orders})

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["user_name", "user_email", "order_id", "order_item", "order_price"])
    writer.writerows(rows)
    input_text = buf.getvalue()
    expected_text = json.dumps(records, indent=2) + "\n"

    meta = {
        "use_case": "uc1_csv_to_json_nested",
        "complexity": complexity,
        "input_format": "csv",
        "output_format": "json",
        "description": (
            "Flat CSV with user_* and order_* columns -> nested JSON with user "
            "object and orders array"
        ),
        "prompt_hint": (
            "Group rows by user (user_name/user_email). user_* fields nest under "
            "'user'; order_* fields become objects in an 'orders' array. Keep "
            "order_id/order_price numeric (do not quote)."
        ),
        "seed": seed,
    }
    return GeneratedCase(
        "uc1_csv_to_json_nested", complexity, "csv", "json", input_text, expected_text, meta
    )
