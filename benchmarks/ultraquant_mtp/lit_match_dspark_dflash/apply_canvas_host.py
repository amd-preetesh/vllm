#!/usr/bin/env python3
"""Host-side: patch lit-match canvas from out/lit_match_qwen38/canvas_rows.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROWS_JSON = Path("/home/preshah/workspace/vllm/out/lit_match_qwen38/canvas_rows.json")
CANVAS = Path(
    "/home/preshah/.cursor/projects/home-preshah-workspace/canvases/lit-match-dspark-dflash.canvas.tsx"
)


def main() -> None:
    if not ROWS_JSON.exists():
        print("missing", ROWS_JSON)
        return
    data = json.loads(ROWS_JSON.read_text())
    rows = data["rows"]
    status = data.get("status", "running")
    refreshed = data.get("refreshed", "")
    text = CANVAS.read_text()
    rows_js = json.dumps(rows, indent=2)
    text2 = re.sub(
        r"const ROWS: Row\[\] = \[[\s\S]*?\];",
        f"const ROWS: Row[] = {rows_js};",
        text,
        count=1,
    )
    text2 = re.sub(
        r'const SUITE_STATUS: "running" \| "done" \| "fail" = "[^"]+";',
        f'const SUITE_STATUS: "running" | "done" | "fail" = "{status}";',
        text2,
        count=1,
    )
    text2 = re.sub(
        r'const REFRESHED = "[^"]+";',
        f'const REFRESHED = "{refreshed}";',
        text2,
        count=1,
    )
    CANVAS.write_text(text2)
    done = sum(1 for r in rows if r.get("tok_s") is not None)
    print(f"updated canvas done={done}/12 status={status}")


if __name__ == "__main__":
    main()
