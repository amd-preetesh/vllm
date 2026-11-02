#!/usr/bin/env python3
"""Aggregate method-match accept + aiperf results into a CSV/JSON summary."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("/workspace/vllm/out")
STEMS = [
    "method_match_dspark_fp8",
    "method_match_dflash_fp8",
    "method_match_dspark_uq_opt",
    "method_match_dflash_uq_swa",
]
OUT_DIR = Path("/workspace/vllm/out/method_match_summary")


def main() -> None:
    rows = []
    for stem in STEMS:
        base = ROOT / stem
        if not base.exists():
            continue
        for metrics in base.glob("*/*/accept_metrics.json"):
            try:
                d = json.loads(metrics.read_text())
            except Exception:  # noqa: BLE001
                continue
            rows.append(d)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / "all_cells.json"
    raw.write_text(json.dumps(rows, indent=2) + "\n")

    # Mean over reps per (arm, workload, conc)
    groups: dict[tuple, list] = {}
    for r in rows:
        key = (r.get("arm"), r.get("workload"), r.get("conc"))
        groups.setdefault(key, []).append(r)

    summary = []
    for (arm, wl, conc), items in sorted(groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]), x[0][2] or 0)):
        toks = [i["output_tok_s"] for i in items if i.get("output_tok_s") is not None]
        rates = [i["accept_rate"] for i in items if i.get("accept_rate") is not None]
        als = [i["accept_length"] for i in items if i.get("accept_length") is not None]
        summary.append(
            {
                "arm": arm,
                "workload": wl,
                "conc": conc,
                "n_reps": len(items),
                "tok_s_mean": statistics.fmean(toks) if toks else None,
                "tok_s_stdev": statistics.stdev(toks) if len(toks) > 1 else 0.0,
                "accept_rate_mean": statistics.fmean(rates) if rates else None,
                "accept_length_mean": statistics.fmean(als) if als else None,
            }
        )

    (OUT_DIR / "summary_by_conc.json").write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = OUT_DIR / "summary_by_conc.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arm",
                "workload",
                "conc",
                "n_reps",
                "tok_s_mean",
                "tok_s_stdev",
                "accept_rate_mean",
                "accept_length_mean",
            ],
        )
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"wrote {len(rows)} cells → {OUT_DIR} ({len(summary)} grouped rows)")


if __name__ == "__main__":
    main()
