#!/usr/bin/env python3
"""Write recovery canvas_rows.json (+ host patch if path visible)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path("/workspace/vllm")
HERE = ROOT / "benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery"
OUT = ROOT / "out/agentx_dspark_dflash_recovery"
HOST_OUT = Path("/home/preshah/workspace/vllm/out/agentx_dspark_dflash_recovery")
CANVAS = Path(
    "/home/preshah/.cursor/projects/home-preshah-workspace/canvases/agentx-dspark-dflash-recovery.canvas.tsx"
)


def main() -> None:
    summary_path = OUT / "summary_gates.json"
    if not summary_path.exists() and HOST_OUT.joinpath("summary_gates.json").exists():
        summary_path = HOST_OUT / "summary_gates.json"
    if not summary_path.exists():
        rows = []
        status = "pending"
    else:
        data = json.loads(summary_path.read_text())
        rows = []
        for r in data.get("rows", []):
            rows.append(
                {
                    "arm": r["arm"],
                    "conc": r["conc"],
                    "tok_s": None if r.get("tok_s") is None else round(float(r["tok_s"]), 1),
                    "accept": None
                    if r.get("accept_length") is None
                    else round(float(r["accept_length"]), 2),
                    "prefix_hit": r.get("prefix_hit"),
                    "vs_nospec": None
                    if r.get("vs_baseline_nospec") is None
                    else round(float(r["vs_baseline_nospec"]), 3),
                }
            )
        status = "done" if data.get("gate_c32", {}).get("pass") else "running"
        if all(r.get("tok_s") is not None for r in rows) and rows:
            status = "done" if data.get("gate_c32", {}).get("pass") else "done"

    pdt = datetime.now(timezone.utc) - timedelta(hours=7)
    refreshed = pdt.strftime("%Y-%m-%d %H:%M PDT")
    payload = {"status": status, "refreshed": refreshed, "rows": rows}
    for base in (OUT, HOST_OUT):
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / "canvas_rows.json").write_text(json.dumps(payload, indent=2) + "\n")
        except Exception:
            pass

    if CANVAS.exists():
        text = CANVAS.read_text()
        rows_js = json.dumps(rows, indent=2)
        text2 = re.sub(
            r"const ROWS: Row\[\] = \[[\s\S]*?\];",
            f"const ROWS: Row[] = {rows_js};",
            text,
            count=1,
        )
        text2 = re.sub(
            r'const SUITE_STATUS: "pending" \| "running" \| "done" = "[^"]+";',
            f'const SUITE_STATUS: "pending" | "running" | "done" = "{status}";',
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
        print(f"updated {CANVAS} rows={len(rows)} status={status}")
    else:
        print("wrote canvas_rows.json only")


if __name__ == "__main__":
    main()
