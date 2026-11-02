#!/usr/bin/env python3
"""Refresh agentx-baseline-live.canvas.tsx + job-dashboard rows from suite outs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/home/preshah/workspace/vllm")
HERE = ROOT / "benchmarks/ultraquant_mtp/agentx_baseline_matrix"
OUT = ROOT / "out/agentx_baseline_qwen38"
ARMS = json.loads((HERE / "arms.json").read_text())
CANVAS = Path(
    "/home/preshah/.cursor/projects/home-preshah-workspace/canvases/"
    "agentx-baseline-live.canvas.tsx"
)
JOB = Path(
    "/home/preshah/.cursor/projects/home-preshah-workspace/canvases/"
    "job-dashboard.canvas.tsx"
)
CONCS = ARMS["concs"]
PDT = ZoneInfo("America/Los_Angeles")


def avg(j: dict | None, key: str) -> float | None:
    if not j:
        return None
    v = j.get(key)
    if isinstance(v, dict):
        a = v.get("avg")
        return float(a) if isinstance(a, (int, float)) else None
    return float(v) if isinstance(v, (int, float)) else None


def cell(arm_id: str, conc: int) -> dict:
    d = OUT / arm_id / "sa" / f"seed42_c{conc}"
    smoke = OUT / arm_id / "smoke" / f"seed42_c{conc}"
    prof = d / "aiperf_artifacts" / "profile_export_aiperf.json"
    st = d / "STATUS"
    if prof.exists():
        j = json.loads(prof.read_text())
        return {
            "status": "done",
            "toks": avg(j, "output_token_throughput"),
            "itl": avg(j, "inter_token_latency"),
        }
    if st.exists():
        s = st.read_text().strip()
        if s == "RUNNING":
            return {"status": "running", "toks": None, "itl": None}
        if s == "FAIL":
            return {"status": "fail", "toks": None, "itl": None}
    if (smoke / "aiperf_artifacts" / "profile_export_aiperf.json").exists() and not d.exists():
        return {"status": "smoke_ok", "toks": None, "itl": None}
    if d.exists() or (d / "server.log").exists():
        return {"status": "partial", "toks": None, "itl": None}
    return {"status": "pending", "toks": None, "itl": None}


def fmt(v: float | None, d: int = 1) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def main() -> None:
    now = datetime.now(PDT).strftime("%Y-%m-%d %H:%M PDT")
    rows = []
    done = running = fail = pending = 0
    total = 0
    table_rows = []
    for arm in ARMS["arms"]:
        aid = arm["id"]
        cells = [cell(aid, c) for c in CONCS]
        n_done = sum(1 for x in cells if x["status"] == "done")
        n_run = sum(1 for x in cells if x["status"] == "running")
        n_fail = sum(1 for x in cells if x["status"] == "fail")
        n_pen = sum(1 for x in cells if x["status"] in ("pending", "partial", "smoke_ok"))
        done += n_done
        running += n_run
        fail += n_fail
        pending += n_pen
        total += len(CONCS)
        toks = [fmt(x["toks"]) for x in cells]
        status = (
            "done"
            if n_done == len(CONCS)
            else "running"
            if n_run
            else "fail"
            if n_fail and not n_done
            else "partial"
            if n_done
            else "pending"
        )
        table_rows.append(
            [
                arm["label"],
                arm["lane"].upper(),
                f"{n_done}/{len(CONCS)}",
                status,
                *toks,
            ]
        )
        rows.append({"id": aid, "label": arm["label"], "status": status, "done": n_done})

    # Build canvas
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    row_lit = ",\n          ".join(
        "[" + ", ".join(json.dumps(c) for c in r) + "]" for r in table_rows
    )
    tone = "success" if done == total and total else "info" if running else "warning"
    canvas = f'''import {{
  Callout,
  Code,
  Divider,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  UsageBar,
}} from "cursor/canvas";

/** Auto-refreshed by agentx_baseline_matrix/refresh_progress.py */
export default function AgentxBaselineLive() {{
  return (
    <Stack gap={{24}}>
      <Stack gap={{6}}>
        <H1>AgentX baseline matrix — Qwen3.8-27B</H1>
        <Text tone="secondary">
          TP=1 · C∈{{8,16,24,32,36}} · HF thinking card (T=1.0, top_p=0.95, top_k=20,
          presence=0, effort=xhigh). FP8 uses <Code>ROCM_AITER_FA</Code>; UQ uses
          UNIFIED / SPEC_COMPAT + UQ draft kernels. Refreshed {esc(now)}.
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone="{tone}">{done}/{total} cells</Pill>
          <Pill tone="info">{running} running</Pill>
          <Pill tone="warning">{fail} fail</Pill>
          <Pill>{pending} pending</Pill>
        </Row>
      </Stack>

      <Row gap={{16}} wrap>
        <Stat value="{done} / {total}" label="AgentX cells done" tone="{tone}" />
        <Stat value="{running}" label="Measuring now" tone="info" />
        <Stat value="{fail}" label="Failed" tone="warning" />
        <Stat value="2" label="GPU lanes (FP8 | UQ)" />
      </Row>

      <UsageBar
        segments={{[
          {{ label: "done", value: {done}, tone: "success" }},
          {{ label: "running", value: {running}, tone: "info" }},
          {{ label: "fail", value: {fail}, tone: "warning" }},
          {{ label: "pending", value: {pending} }},
        ]}}
      />

      <H2>Throughput by arm (tok/s)</H2>
      <Table
        headers={{[
          "Arm",
          "Lane",
          "Progress",
          "Status",
          "C8",
          "C16",
          "C24",
          "C32",
          "C36",
        ]}}
        columnAlign={{[
          "left",
          "left",
          "right",
          "left",
          "right",
          "right",
          "right",
          "right",
          "right",
        ]}}
        rows={{[
          {row_lit}
        ]}}
        striped
      />

      <Callout tone="info" title="Suite layout">
        Outs under <Code>vllm/out/agentx_baseline_qwen38/&lt;arm&gt;/sa/</Code>.
        Smoke gate = nospec c8 @ 60s. Full = 300s AgentX per cell. UQ MTP prod =
        FlyDSL PDV off; UQ* MTP = PDV1.1 shipping verify.
      </Callout>

      <Divider />
      <Text size="small" tone="tertiary">
        Manifest: <Code>benchmarks/ultraquant_mtp/agentx_baseline_matrix/arms.json</Code>
      </Text>
    </Stack>
  );
}}
'''
    CANVAS.write_text(canvas)
    print(f"wrote {CANVAS} done={done}/{total} running={running}")

    # Light-touch job dashboard header rewrite if markers present
    if JOB.exists():
        text = JOB.read_text()
        # Only update the snapshot line if present
        import re

        text2 = re.sub(
            r"Snapshot [^.]+\.",
            f"Snapshot {now}. Focus: AgentX baseline matrix ({done}/{total}).",
            text,
            count=1,
        )
        if text2 != text:
            JOB.write_text(text2)
            print(f"touched {JOB}")


if __name__ == "__main__":
    main()
