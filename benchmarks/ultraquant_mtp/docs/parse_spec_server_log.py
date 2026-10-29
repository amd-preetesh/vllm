#!/usr/bin/env python3
"""Summarize vLLM server.log stat lines for spec-decode AgentX cells.

Usage: parse_spec_server_log.py server.log [server.log ...] [--series]
"""
import re
import sys

STAT = re.compile(
    r"(\d\d-\d\d \d\d:\d\d:\d\d).*Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, "
    r"Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, "
    r"Prefix cache hit rate: ([\d.]+)%")
SPEC = re.compile(
    r"(\d\d-\d\d \d\d:\d\d:\d\d).*SpecDecoding metrics: Mean acceptance length: "
    r"([\d.]+).*?Accepted throughput: ([\d.]+) tokens/s, Drafted throughput: "
    r"([\d.]+) tokens/s.*?Per-position acceptance rate: ([\d., ]+)")
PREEMPT = re.compile(r"preempt", re.I)


def summarize(path, series):
    stats, specs, npre = [], [], 0
    with open(path, errors="replace") as f:
        for line in f:
            m = STAT.search(line)
            if m:
                stats.append((m.group(1), *map(float, m.groups()[1:])))
                continue
            m = SPEC.search(line)
            if m:
                specs.append((m.group(1), float(m.group(2)), float(m.group(3)),
                              float(m.group(4)), m.group(5).strip()))
                continue
            if PREEMPT.search(line):
                npre += 1
    print(f"=== {path}")
    if not stats:
        print("  no stat lines")
        return
    busy = [s for s in stats if s[4] > 0 or s[2] > 0]
    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    print(f"  stat lines={len(stats)} busy={len(busy)} preempt-lines={npre}")
    if busy:
        print("  busy avg: prompt_tps=%.0f gen_tps=%.1f running=%.1f waiting=%.1f "
              "kv%%=%.1f max_kv%%=%.1f max_wait=%d prefix_hit%%(last)=%.1f" % (
                  avg([s[1] for s in busy]), avg([s[2] for s in busy]),
                  avg([s[3] for s in busy]), avg([s[4] for s in busy]),
                  avg([s[5] for s in busy]), max(s[5] for s in busy),
                  max(s[4] for s in busy), busy[-1][6]))
    if specs:
        print("  spec lines=%d mean_accept_len avg=%.2f min=%.2f max=%.2f" % (
            len(specs), avg([s[1] for s in specs]), min(s[1] for s in specs),
            max(s[1] for s in specs)))
        print("  last per-pos accept:", specs[-1][4])
    if series:
        for s in stats:
            print("   %s prompt=%7.0f gen=%7.1f run=%3d wait=%3d kv=%5.1f hit=%5.1f" % s)
        for s in specs:
            print("   %s acc_len=%.2f acc_tps=%.1f draft_tps=%.1f pos=%s" % s)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for p in args:
        summarize(p, "--series" in sys.argv)
