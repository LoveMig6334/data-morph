# Showcase

Three messy, real-world files converted **end to end by the 2.0 GB student** — no
hand-written rules. For each case you'll see the **input**, the **Python script the model
wrote** from the file's metadata envelope, the **converted output**, and its **validation
scores**.

All three ran through the public API exactly as you would:

```python
from datamorph import convert_file
convert_file("orders.csv", "orders.json", instruction="...")
```

These are **live model outputs** (greedy decoding, so reproducible), not curated by hand.
Notice the model picks up real-world details on its own — stripping a BOM, grouping rows
into a nested array, flattening nested keys to dot-notation, and falling back gracefully on
a log line that doesn't match the expected shape.

## 1. Ragged CSV → nested JSON

**Input** — `orders.csv`:

```csv
user_name,user_email,order_id,order_item,order_qty
Alice,alice@example.com,1001,Widget,3
Alice,alice@example.com,1002,Gadget,1
Bob,bob@example.com,1003,Gizmo,5
```

**The student wrote this script** (retries: 1):

```python
import sys
import json
import pandas as pd


def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    # Envelope encoding is utf-8-sig; read with it to strip the BOM.
    df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)

    # NUMERIC_COLUMN_QUOTE_RISK: keep these numeric in JSON, never stringify.
    for col in ("order_id", "order_qty"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    def clean(v):
        if v is None:
            return ""
        if isinstance(v, float) and pd.isna(v):
            return ""
        if isinstance(v, str):
            return v.strip()
        return str(v)

    users = []
    index = {}  # (user_name, user_email) -> position in `users`

    for _, row in df.iterrows():
        name = clean(row.get("user_name"))
        email = clean(row.get("user_email"))
        key = (name, email)

        if key not in index:
            index[key] = len(users)
            users.append({
                "user": {"name": name, "email": email},
                "orders": [],
            })

        oid = row.get("order_id")
        oitem = clean(row.get("order_item"))
        oq = row.get("order_qty")
        users[index[key]]["orders"].append({
            "id": int(oid),
            "item": oitem,
            "q": int(oq),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
```

**Converted output** — `orders.json`:

```json
[
  {
    "user": {
      "name": "Alice",
      "email": "alice@example.com"
    },
    "orders": [
      {
        "id": 1001,
        "item": "Widget",
        "q": 3
      },
      {
        "id": 1002,
        "item": "Gadget",
        "q": 1
      }
    ]
  },
  {
    "user": {
      "name": "Bob",
      "email": "bob@example.com"
    },
    "orders": [
      {
        "id": 1003,
        "item": "Gizmo",
        "q": 5
      }
    ]
  }
]
```

!!! success "✓ accepted"
    format_validity 1.0 · loadability 1.0

## 2. Nested JSON (optional keys) → CSV

**Input** — `users.json`:

```json
[
  {"id": 1, "name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
  {"id": 2, "name": "Bob", "address": {"city": "Boston"}},
  {"id": 3, "name": "Carol", "address": {"city": "LA", "zip": "90001"}}
]
```

**The student wrote this script** (retries: 1):

```python
import sys
import json
import pandas as pd


def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Envelope says root_shape is a record_array; tolerate a single object too.
    if isinstance(data, dict):
        data = [data]

    # Flatten nested keys into dot-notation columns (address.city, address.zip).
    df = pd.json_normalize(data, sep=".")

    # NUMERIC_COLUMN_QUOTE_RISK: keep id numeric so it is not stringified.
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

    # Keep address.zip as a string so leading zeros are preserved; coerce others.
    df["address.zip"] = df["address.zip"].astype("string")

    # Handle missing values (nulls, missing keys) by leaving them empty.
    df.to_csv(out_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
```

**Converted output** — `users.csv`:

```csv
id,name,address.city,address.zip
1,Alice,NYC,10001
2,Bob,Boston,
3,Carol,LA,90001
```

!!! success "✓ accepted"
    format_validity 1.0 · loadability 1.0

## 3. Semi-structured TXT log → CSV

**Input** — `events.log`:

```text
[2026-01-05 08:12:01] INFO auth: user alice logged in
[2026-01-05 08:13:44] WARNING auth: failed login for bob
[2026-01-05 08:15:02] ERROR db: connection timeout
```

**The student wrote this script** (retries: 0):

```python
import sys
import re
from datetime import datetime

import pandas as pd

LINE_RE = re.compile(
    r"^\[?(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})\]?\s+"
    r"(\S+)\s+"
    r"([^:]+):\s?"
    r"(.*)$"
)


def to_iso(date_str, time_str):
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return dt.isoformat()


def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    rows = []
    with open(in_path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            m = LINE_RE.match(line)
            if m:
                date_str, time_str, level, source, message = m.groups()
                rows.append(
                    {
                        "timestamp": to_iso(date_str, time_str),
                        "level": level,
                        "source": source,
                        "message": message,
                    }
                )
            else:
                # Non-conforming line: preserve content, leave level/source empty.
                rows.append(
                    {"timestamp": "", "level": "", "source": "", "message": line.strip()}
                )

    df = pd.DataFrame(rows, columns=["timestamp", "level", "source", "message"])
    df.to_csv(out_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
```

**Converted output** — `events.csv`:

```csv
timestamp,level,source,message
2026-01-05T08:12:01,INFO,auth,user alice logged in
2026-01-05T08:13:44,WARNING,auth,failed login for bob
2026-01-05T08:15:02,ERROR,db,connection timeout
```

!!! success "✓ accepted"
    format_validity 1.0 · loadability 1.0
