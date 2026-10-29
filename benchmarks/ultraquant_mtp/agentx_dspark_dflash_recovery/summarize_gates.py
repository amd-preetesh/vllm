#!/usr/bin/env python3
"""Summarize AgentX DSpark/DFlash recovery vs baseline nospec and gates."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/workspace/vllm")
HERE = ROOT / "benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery"
OUT = ROOT / "out/agentx_dspark_dflash_recovery"
BASE = ROOT / "out/agentx_baseline_qwen38"
ARMS = json.loads((HERE / "arms.json").read_text())
GATES = ARMS["pass_gates"]


def toks(path: Path) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    v = d.get("output_token_throughput", {})
    if isinstance(v, dict):
        return float(v.get("avg")) if v.get("avg") is not None else None
    return None


def accept_len(metrics: Path) -> float | None:
    if not metrics.exists():
        return None
    text = metrics.read_text(errors="replace")

    def grab(name: str) -> float | None:
        total = 0.0
        found = False
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if not line.startswith(f"vllm:{name}"):
                continue
            metric = line.split("{", 1)[0]
            if metric != f"vllm:{name}":
                continue
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    total += float(parts[1])
                    found = True
                except Exception:
                    pass
        return total if found else None

    a = grab("spec_decode_num_accepted_tokens_total")
    n = grab("spec_decode_num_drafts_total")
    if a is None or n is None or n <= 0:
        return None
    return 1.0 + a / n


def prefix_hit(metrics: Path) -> float | None:
    if not metrics.exists():
        return None
    text = metrics.read_text(errors="replace")

    def grab(name: str) -> float | None:
        total = 0.0
        found = False
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if not line.startswith(f"vllm:{name}"):
                continue
            metric = line.split("{", 1)[0]
            if metric != f"vllm:{name}":
                continue
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    total += float(parts[1])
                    found = True
                except Exception:
                    pass
        return total if found else None

    h = grab("prefix_cache_hits_total")
    q = grab("prefix_cache_queries_total")
    if h is None or q is None or q <= 0:
        return None
    return h / q


def main() -> None:
    rows = []
    for a in ARMS["arms"]:
        aid = a["id"]
        lane = a["lane"]
        nospec_id = f"{lane}_nospec"
        for c in ARMS["concs"]:
            cell = OUT / aid / "sa" / f"seed42_c{c}"
            prof = cell / "aiperf_artifacts/profile_export_aiperf.json"
            met = cell / "metrics_end.txt"
            t = toks(prof)
            al = accept_len(met)
            hit = prefix_hit(met)
            base_prof = (
                BASE / nospec_id / "sa" / f"seed42_c{c}" / "aiperf_artifacts/profile_export_aiperf.json"
            )
            bt = toks(base_prof)
            ratio = (t / bt) if (t and bt) else None
            rows.append(
                {
                    "arm": aid,
                    "conc": c,
                    "tok_s": t,
                    "accept_length": al,
                    "prefix_hit": hit,
                    "vs_baseline_nospec": ratio,
                }
            )

    # Gate check at C32
    gate = {"pass": True, "arms": {}}
    for a in ARMS["arms"]:
        aid = a["id"]
        row = next((r for r in rows if r["arm"] == aid and r["conc"] == 32), None)
        if not row or row["tok_s"] is None:
            gate["arms"][aid] = {"pass": False, "reason": "missing"}
            gate["pass"] = False
            continue
        ok_speed = (row["vs_baseline_nospec"] or 0) >= GATES["vs_nospec_c32"]
        ok_acc = (row["accept_length"] or 0) >= GATES["accept_min"]
        ok_hit = (row["prefix_hit"] or 0) >= GATES["prefix_hit_min"]
        gate["arms"][aid] = {
            "pass": bool(ok_speed and ok_acc and ok_hit),
            "tok_s": row["tok_s"],
            "vs_nospec": row["vs_baseline_nospec"],
            "accept_length": row["accept_length"],
            "prefix_hit": row["prefix_hit"],
        }
        if not gate["arms"][aid]["pass"]:
            gate["pass"] = False

    out = {"rows": rows, "gate_c32": gate}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary_gates.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("PASS" if gate["pass"] else "FAIL")


if __name__ == "__main__":
    main()
