#!/usr/bin/env python3
"""Refresh dspark-dflash-recovery-progress.canvas.tsx from suite outs (host)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

WS = Path("/home/preshah/workspace")
OUT = WS / "vllm/out/lit_match_qwen38"
STATUS = WS / "vllm/out/recovery_progress.json"
# container may write under /workspace which is the same bind on this host
STATUS_ALT = Path("/home/preshah/workspace/vllm/out/recovery_progress.json")
CANVAS = Path(
    "/home/preshah/.cursor/projects/home-preshah-workspace/canvases/dspark-dflash-recovery-progress.canvas.tsx"
)
ARMS = json.loads(
    (WS / "vllm/benchmarks/ultraquant_mtp/lit_match_dspark_dflash/arms.json").read_text()
)
SUITE_DONE = Path("/home/preshah/workspace/logs/lit_match_qwen38/suite.DONE")
HANDOFF_DONE = Path("/home/preshah/workspace/logs/dspark_dflash_handoff/handoff.DONE")


def lit_rows():
    rows = []
    by = {}
    for a in ARMS["arms"]:
        for c in ARMS["concs"]:
            am = OUT / a["id"] / f"c{c}_r1" / "accept_metrics.json"
            if am.exists():
                by[(a["id"], c)] = json.loads(am.read_text())
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
                need = ARMS["pass_gates"][f"c{c}"][method]
                acc_need = ARMS["pass_gates"]["accept_min"][method]
                if vs < need or acc is None or float(acc) < acc_need:
                    gate = "fail"
            elif a["id"] in ("fp8_dspark", "fp8_dflash"):
                gate = "pending"
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
    return rows


def current_cell_hint() -> str:
    # Prefer RUNNING STATUS dirs
    for a in ARMS["arms"]:
        for c in ARMS["concs"]:
            st = OUT / a["id"] / f"c{c}_r1" / "STATUS"
            if st.exists() and st.read_text().strip() == "RUNNING":
                return f"{a['id']} @C{c} RUNNING on HIP3"
    return "waiting for next cell / handoff"


def main() -> None:
    rows = lit_rows()
    done = sum(1 for r in rows if r["tok_s"] is not None)
    lit_pass = None
    sg = OUT / "summary_gates.json"
    if done == 12 and sg.exists():
        try:
            lit_pass = bool(json.loads(sg.read_text()).get("pass"))
        except Exception:
            lit_pass = None

    status = {"phase": "phase1_running", "detail": current_cell_hint(), "hip": 3}
    for sp in (STATUS, STATUS_ALT):
        if sp.exists():
            try:
                status.update(json.loads(sp.read_text()))
            except Exception:
                pass
            break
    if HANDOFF_DONE.exists():
        status["phase"] = "done"
        status["detail"] = "handoff.DONE — Phase 1+2 finished"
    elif SUITE_DONE.exists() and status.get("phase") in (None, "wait_phase1", "phase1_running"):
        status["phase"] = status.get("phase") or "phase2_swa_probe"
        if status["phase"] == "phase1_running":
            status["phase"] = "phase2_swa_probe"
            status["detail"] = "suite.DONE seen — Phase 2 starting"

    phase = status.get("phase") or "phase1_running"
    detail = status.get("detail") or current_cell_hint()
    hip = int(status.get("hip") or 3)
    pdt = datetime.now(timezone.utc) - timedelta(hours=7)
    refreshed = pdt.strftime("%Y-%m-%d %H:%M PDT")

    lit_pass_js = "null" if lit_pass is None else ("true" if lit_pass else "false")
    # ASCII-only detail for safe regex replacement
    detail_s = detail.replace("\u2014", "-").replace('"', "'")
    text = CANVAS.read_text()

    def sub_once(pat: str, repl: str, src: str) -> str:
        out, n = re.subn(pat, lambda m: repl, src, count=1)
        if n != 1:
            print("warn: no match", pat[:50])
        return out

    text2 = text
    text2 = sub_once(
        r'const PHASE:\n  \| "wait_phase1"\n  \| "phase1_running"\n  \| "phase2_swa_probe"\n  \| "phase2_agentx"\n  \| "done" = "[^"]+";',
        'const PHASE:\n  | "wait_phase1"\n  | "phase1_running"\n  | "phase2_swa_probe"\n  | "phase2_agentx"\n  | "done" = "%s";'
        % phase,
        text2,
    )
    text2 = sub_once(r'const DETAIL = "[^"]*";', f'const DETAIL = "{detail_s}";', text2)
    text2 = sub_once(r"const HIP = \d+;", f"const HIP = {hip};", text2)
    text2 = sub_once(r"const LIT_DONE = \d+;", f"const LIT_DONE = {done};", text2)
    text2 = sub_once(
        r"const LIT_PASS: boolean \| null = [^;]+;",
        f"const LIT_PASS: boolean | null = {lit_pass_js};",
        text2,
    )
    text2 = sub_once(r'const REFRESHED = "[^"]*";', f'const REFRESHED = "{refreshed}";', text2)
    text2 = sub_once(
        r"const LIT_ROWS: LitRow\[\] = \[[\s\S]*?\];",
        "const LIT_ROWS: LitRow[] = " + json.dumps(rows, indent=2) + ";",
        text2,
    )
    CANVAS.write_text(text2)
    print(f"progress canvas phase={phase} lit={done}/12")


if __name__ == "__main__":
    main()
