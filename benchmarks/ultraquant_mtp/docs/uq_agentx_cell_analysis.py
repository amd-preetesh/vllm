#!/usr/bin/env python3
"""Read-only AgentX cell analysis used by uq_dflash_dspark_bottlenecks.md.

Everything is derived from files that aiperf / the sweep already wrote:
  <cell>/aiperf_artifacts/{profile_export_aiperf.json, server_metrics_export.json,
                           profile_export.jsonl}, <cell>/server.log

Subcommands (all take one or more cell dirs, relative to vllm/out or absolute):
  table   profile-phase summary: tok/s (aiperf and server-side), ITL, TTFT, hit,
          theoretical hit, accept, KV%, miss tokens/request, steps
  events  per-request prefix-cache events (queries/hits per 1 s timeslice) and the
          zero-hit / partial / good split, with miss-token share per class
  step    median engine step time per running-count from server.log spec lines
  accept  per-position acceptance over time from server.log (flags zero positions)

Example:
  uq_agentx_cell_analysis.py table dspark_qwen38_uq/sa/seed42_c32 \
      pdv_stream1_sa/qwen3-8-27b_ultraquant_pdv_mtp5_tp1_c32
"""
import json
import os
import re
import statistics as st
import sys

OUT = os.environ.get("VLLM_OUT", "/home/preshah/workspace/vllm/out")


def _p(cell):
    return cell if os.path.isabs(cell) else os.path.join(OUT, cell)


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _series_stat(m, key, field="total"):
    if key not in m:
        return None
    tot, found = 0.0, False
    for s in m[key]["series"]:
        v = s["stats"].get(field)
        if v is not None:
            tot += v
            found = True
    return tot if found else None


def _timeslices(m, key):
    s = m[key]["series"][0]["timeslices"]
    return {x["start_ns"]: x["total"] for x in s}


def cell_summary(cell):
    base = _p(cell)
    art = base + "/aiperf_artifacts"
    pe = _load(art + "/profile_export_aiperf.json")
    se = _load(art + "/server_metrics_export.json")
    if not pe or not se:
        return None
    g = lambda k, f="avg": (pe.get(k) or {}).get(f)
    m = se["metrics"]
    pr = se["summary"]["phase_time_ranges"]["profiling"]
    dur = (pr["end_ns"] - pr["start_ns"]) / 1e9
    r = dict(dur=dur)
    r["n_done"] = g("request_count")
    r["tok_s"] = g("output_token_throughput")
    r["itl_avg"] = g("inter_token_latency")
    r["itl_p50"] = g("inter_token_latency", "p50")
    r["ttft_avg_s"] = (g("time_to_first_token") or 0) / 1000.0
    r["isl"] = g("input_sequence_length")
    r["osl"] = g("output_sequence_length")
    r["theo_hit"] = g("theoretical_prefix_cache_hit")
    h = _series_stat(m, "vllm:prefix_cache_hits")
    q = _series_stat(m, "vllm:prefix_cache_queries")
    r["hit"] = (h / q) if q else None
    r["queries"], r["hits"] = q, h
    r["kv_avg"] = _series_stat(m, "vllm:kv_cache_usage_perc", "avg")
    r["kv_max"] = _series_stat(m, "vllm:kv_cache_usage_perc", "max")
    r["run_avg"] = _series_stat(m, "vllm:num_requests_running", "avg")
    r["wait_avg"] = _series_stat(m, "vllm:num_requests_waiting", "avg")
    r["preempt"] = _series_stat(m, "vllm:num_preemptions")
    d = _series_stat(m, "vllm:spec_decode_num_drafts")
    a = _series_stat(m, "vllm:spec_decode_num_accepted_tokens")
    r["acc_len"] = (1 + a / d) if d else None
    r["gen_tok"] = _series_stat(m, "vllm:generation_tokens")
    r["srv_tok_s"] = r["gen_tok"] / dur if r["gen_tok"] is not None else None
    try:
        r["steps"] = m["vllm:iteration_tokens_total"]["series"][0]["stats"]["count"]
    except Exception:
        r["steps"] = None
    if q and r["isl"]:
        r["reqs_in_window"] = q / r["isl"]
        r["miss_per_req_k"] = (q - h) / r["reqs_in_window"] / 1e3
    return r


def cmd_table(cells):
    hdr = ("cell", "done", "aip tok/s", "srv tok/s", "ITL avg", "ITL p50",
           "TTFT s", "hit%", "theo%", "acc", "KV avg/max%", "run", "wait",
           "steps", "miss k/req")
    print(" | ".join(hdr))
    f = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"
    for c in cells:
        r = cell_summary(c)
        if not r:
            print(c, "| MISSING")
            continue
        print(" | ".join([
            c, f(r["n_done"], 0), f(r["tok_s"]), f(r["srv_tok_s"]),
            f(r["itl_avg"]), f(r["itl_p50"]), f(r["ttft_avg_s"]),
            f(r["hit"] * 100 if r["hit"] is not None else None),
            f(r["theo_hit"]), f(r["acc_len"], 2),
            f"{f(r['kv_avg'] * 100, 0)}/{f(r['kv_max'] * 100, 0)}",
            f(r["run_avg"]), f(r["wait_avg"], 2), f(r["steps"], 0),
            f(r.get("miss_per_req_k"), 0)]))


def cell_events(cell, min_q=20000):
    se = _load(_p(cell) + "/aiperf_artifacts/server_metrics_export.json")
    m = se["metrics"]
    h = _timeslices(m, "vllm:prefix_cache_hits")
    q = _timeslices(m, "vllm:prefix_cache_queries")
    return [(k, q[k], h.get(k, 0)) for k in sorted(q) if q[k] > min_q]


def cmd_events(cells, verbose=False):
    for c in cells:
        if not _load(_p(c) + "/aiperf_artifacts/server_metrics_export.json"):
            print(f"{c}: MISSING / not finished")
            continue
        ev = cell_events(c)
        zero = [(q, h) for _, q, h in ev if h / q < 0.05]
        part = [(q, h) for _, q, h in ev if 0.05 <= h / q < 0.6]
        good = [(q, h) for _, q, h in ev if h / q >= 0.6]
        tm = sum(q - h for _, q, h in ev) or 1
        zm = sum(q - h for q, h in zero)
        gm = [q - h for q, h in good]
        print(f"{c}: events={len(ev)} zero-hit={len(zero)} partial={len(part)} "
              f"good={len(good)} | zero-hit share of miss tokens={zm / tm:.0%} | "
              f"good-hit mean miss={st.mean(gm) / 1e3 if gm else 0:.1f}k "
              f"median={st.median(gm) / 1e3 if gm else 0:.1f}k")
        if verbose:
            t0 = ev[0][0] if ev else 0
            for k, q, h in ev:
                print(f"   +{(k - t0) / 1e9:5.0f}s q={q:8.0f} hit={h:8.0f} "
                      f"({100 * h / q:5.1f}%) miss={q - h:8.0f}")


STAT = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d).*Avg prompt throughput: ([\d.]+) tokens/s, "
                  r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, "
                  r"Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")
SPEC = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d).*Mean acceptance length: ([\d.]+).*?"
                  r"Accepted throughput: ([\d.]+) tokens/s, Drafted throughput: ([\d.]+) "
                  r"tokens/s, Accepted: (\d+) tokens, Drafted: (\d+) tokens, "
                  r"Per-position acceptance rate: ([\d., ]+)")


def _parse_log(cell):
    stats, specs = {}, {}
    with open(_p(cell) + "/server.log", errors="replace") as fh:
        for line in fh:
            m = STAT.search(line)
            if m:
                stats[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                     int(m.group(4)), int(m.group(5)),
                                     float(m.group(6)))
                continue
            m = SPEC.search(line)
            if m:
                specs[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                     float(m.group(4)), int(m.group(5)),
                                     int(m.group(6)),
                                     [float(x) for x in m.group(7).strip(", ").split(",")])
    return stats, specs


def cmd_step(cells, k):
    """Engine-step proxy: 1000 / (drafted_tps / K) is ms per request-step; the
    engine step is that times the running count (all running reqs decode)."""
    for c in cells:
        stats, specs = _parse_log(c)
        by_run = {}
        for t, s in stats.items():
            sp = specs.get(t)
            if not sp or sp[4] < k * 40:
                continue
            ms_req = 1000.0 / (sp[2] / k)
            by_run.setdefault(s[2], []).append((ms_req, sp[0]))
        parts = []
        for run in sorted(by_run):
            v = by_run[run]
            if len(v) >= 2:
                parts.append(f"run={run}: {st.median(x[0] for x in v) * run:.0f}ms/step "
                             f"(n={len(v)}, acc {st.median(x[1] for x in v):.2f})")
        print(c, "|", " ; ".join(parts[:6]))


def cmd_accept(cells):
    for c in cells:
        stats, specs = _parse_log(c)
        print("==", c)
        for t in sorted(specs):
            sp = specs[t]
            pos = sp[5]
            flag = ""
            if sp[4] >= 70 and len(pos) >= 4 and pos[0] > 0.3 and max(pos[2:]) < 0.01:
                flag = "  <== pos>=2 all zero"
            elif sp[4] >= 70 and max(pos) < 0.01:
                flag = "  <== all zero"
            run = stats.get(t, (0, 0, "?", 0, 0))[2]
            print(f"  {t} run={run} acc={sp[0]:.2f} drafted={sp[4]} "
                  f"pos={','.join(f'{x:.2f}' for x in pos)}{flag}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "table":
        cmd_table(args)
    elif cmd == "events":
        cmd_events([a for a in args if a != "-v"], verbose="-v" in args)
    elif cmd == "step":
        cmd_step(args, k=int(os.environ.get("K", "7")))
    elif cmd == "accept":
        cmd_accept(args)
    else:
        print(__doc__)
        sys.exit(1)
