#!/usr/bin/env python3
"""Summarize lit_match_qwen38 results and evaluate Phase-1 pass gates."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/workspace/vllm")
HERE = ROOT / "benchmarks/ultraquant_mtp/lit_match_dspark_dflash"
OUT = ROOT / "out/lit_match_qwen38"
ARMS = json.loads((HERE / "arms.json").read_text())
GATES = ARMS["pass_gates"]
CONCS = ARMS["concs"]
ARM_IDS = [a["id"] for a in ARMS["arms"]]


def load_cell(arm: str, conc: int) -> dict | None:
    d = OUT / arm / f"c{conc}_r1"
    am = d / "accept_metrics.json"
    if not am.exists():
        return None
    return json.loads(am.read_text())


def main() -> None:
    rows = []
    by = {}
    for arm in ARM_IDS:
        for c in CONCS:
            cell = load_cell(arm, c)
            if not cell:
                rows.append({"arm": arm, "conc": c, "status": "missing"})
                continue
            by[(arm, c)] = cell
            rows.append(
                {
                    "arm": arm,
                    "conc": c,
                    "status": "ok",
                    "tok_s": cell.get("output_tok_s"),
                    "accept_length": cell.get("accept_length"),
                }
            )

    summary = {"rows": rows, "gates": {}, "pass": True}
    for c in CONCS:
        nospec = by.get(("fp8_nospec", c))
        if not nospec or not nospec.get("output_tok_s"):
            summary["gates"][f"c{c}"] = {"pass": False, "reason": "nospec missing"}
            summary["pass"] = False
            continue
        base = float(nospec["output_tok_s"])
        gkey = f"c{c}"
        gate = GATES[gkey]
        entry = {"nospec_tok_s": base, "arms": {}}
        ok = True
        for method, need in gate.items():
            arm = f"fp8_{method}" if method != "dflash" else "fp8_dflash"
            # gate keys are dspark/dflash
            arm = {"dspark": "fp8_dspark", "dflash": "fp8_dflash"}[method]
            cell = by.get((arm, c))
            if not cell or not cell.get("output_tok_s"):
                entry["arms"][arm] = {"pass": False, "reason": "missing"}
                ok = False
                continue
            toks = float(cell["output_tok_s"])
            ratio = toks / base if base else 0.0
            acc = cell.get("accept_length")
            acc_need = GATES["accept_min"][method]
            speed_ok = ratio >= float(need)
            acc_ok = (acc is not None) and (float(acc) >= float(acc_need))
            entry["arms"][arm] = {
                "tok_s": toks,
                "vs_nospec": round(ratio, 3),
                "accept_length": acc,
                "speed_gate": need,
                "accept_gate": acc_need,
                "pass": bool(speed_ok and acc_ok),
            }
            if not (speed_ok and acc_ok):
                ok = False
        entry["pass"] = ok
        summary["gates"][gkey] = entry
        if not ok:
            summary["pass"] = False

    out_path = OUT / "summary_gates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("PASS" if summary["pass"] else "FAIL", "->", out_path)


if __name__ == "__main__":
    main()
