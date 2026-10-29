#!/usr/bin/env python3
"""Rewrite lit-match-dspark-dflash.canvas.tsx ROWS from out/lit_match_qwen38."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path("/workspace/vllm")
HERE = ROOT / "benchmarks/ultraquant_mtp/lit_match_dspark_dflash"
OUT = ROOT / "out/lit_match_qwen38"
ARMS = json.loads((HERE / "arms.json").read_text())
CANVAS = Path(
    "/home/preshah/.cursor/projects/home-preshah-workspace/canvases/lit-match-dspark-dflash.canvas.tsx"
)
# Host path when script runs inside container may differ — try both.
HOST_CANVAS = Path(
    "/home/preshah/.cursor/projects/home-preshah-workspace/canvases/lit-match-dspark-dflash.canvas.tsx"
)
CONT_CANVAS = Path(
    "/workspace/../.cursor/projects/home-preshah-workspace/canvases/lit-match-dspark-dflash.canvas.tsx"
)


def load_cell(arm: str, conc: int) -> dict | None:
    am = OUT / arm / f"c{conc}_r1" / "accept_metrics.json"
    if not am.exists():
        return None
    return json.loads(am.read_text())


def main() -> None:
    by = {}
    for a in ARMS["arms"]:
        for c in ARMS["concs"]:
            cell = load_cell(a["id"], c)
            if cell:
                by[(a["id"], c)] = cell

    rows = []
    gates = ARMS["pass_gates"]
    for a in ARMS["arms"]:
        for c in ARMS["concs"]:
            cell = by.get((a["id"], c))
            nospec = by.get(("fp8_nospec", c))
            if not cell:
                rows.append(
                    {
                        "arm": a["id"],
                        "conc": c,
                        "tok_s": None,
                        "accept": None,
                        "vs_nospec": None,
                        "gate": "pending",
                    }
                )
                continue
            toks = cell.get("output_tok_s")
            acc = cell.get("accept_length")
            vs = None
            if toks is not None and nospec and nospec.get("output_tok_s"):
                vs = float(toks) / float(nospec["output_tok_s"])
            gate = "pass"
            if a["id"] in ("fp8_dspark", "fp8_dflash") and vs is not None:
                method = "dspark" if "dspark" in a["id"] else "dflash"
                need = gates[f"c{c}"][method]
                acc_need = gates["accept_min"][method]
                if vs < need or (acc is None) or float(acc) < acc_need:
                    gate = "fail"
            elif a["id"] in ("fp8_dspark", "fp8_dflash"):
                gate = "pending"
            else:
                gate = "pass" if toks is not None else "pending"
            rows.append(
                {
                    "arm": a["id"],
                    "conc": c,
                    "tok_s": None if toks is None else round(float(toks), 1),
                    "accept": None if acc is None else round(float(acc), 2),
                    "vs_nospec": None if vs is None else round(vs, 3),
                    "gate": gate,
                }
            )

    done = sum(1 for r in rows if r["tok_s"] is not None)
    status = "done" if done == 12 else ("fail" if (OUT / "summary_gates.json").exists() and not json.loads((OUT / "summary_gates.json").read_text()).get("pass", True) and done == 12 else "running")
    if done == 12 and (OUT / "summary_gates.json").exists():
        status = "done" if json.loads((OUT / "summary_gates.json").read_text()).get("pass") else "fail"
    elif done == 12:
        status = "done"
    pdt = datetime.now(timezone.utc) - timedelta(hours=7)
    refreshed = pdt.strftime("%Y-%m-%d %H:%M PDT")

    payload = {"status": status, "refreshed": refreshed, "rows": rows}
    (OUT / "canvas_rows.json").write_text(json.dumps(payload, indent=2) + "\n")

    # Prefer rewriting via a data sidecar pattern: patch the const ROWS block.
    target = HOST_CANVAS if HOST_CANVAS.exists() else (CONT_CANVAS if CONT_CANVAS.exists() else None)
    if target is None:
        print("wrote", OUT / "canvas_rows.json", f"done={done}/12")
        return

    import re

    text = target.read_text()
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
    target.write_text(text2)
    print(f"updated {target} done={done}/12 status={status}")


if __name__ == "__main__":
    main()
