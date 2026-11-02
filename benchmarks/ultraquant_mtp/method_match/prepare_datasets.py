#!/usr/bin/env python3
"""Prepare Spec-Bench + SPEED-Bench throughput-1K balanced-747 for method-match.

SPEED-Bench procedure mirrors the MI350X four-draft study:
  - start from nvidia/SPEED-Bench throughput_1k (512/512/512)
  - drop unresolved HLE-derived mixed rows (263 gated placeholders)
  - resolve remaining AdaLEval / BAMBOO placeholders when possible
  - source-stratified largest-remainder subsample of low/high to 249 each
    (seed 20260911); keep 249 resolvable mixed → 747 total

Writes:
  datasets/spec_bench_first_turn.jsonl
  datasets/speed_bench_throughput_1k_balanced_747.jsonl
  datasets/MANIFEST.json
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
OUT = HERE / "datasets"
SEED = 20260911
PH = "FULL BENCHMARK DATA SHOULD BE FETCHED"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_spec_bench() -> Path:
    url = (
        "https://raw.githubusercontent.com/hemingkx/Spec-Bench/"
        "fd2c1cd7d2201ef71db4c5f4e455008f017967bf/data/spec_bench/question.jsonl"
    )
    raw = urllib.request.urlopen(url, timeout=120).read().decode("utf-8")
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        turns = obj.get("turns") or []
        if not turns:
            continue
        prompt = str(turns[0]).strip()
        if not prompt:
            continue
        rows.append(
            {
                "session_id": str(obj.get("question_id", len(rows))),
                "turns": [{"text": prompt}],
            }
        )
    out = OUT / "spec_bench_first_turn.jsonl"
    write_jsonl(out, rows)
    return out


def _largest_remainder(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values()) or 1
    raw = {k: target * v / total for k, v in counts.items()}
    floors = {k: int(math.floor(v)) for k, v in raw.items()}
    rem = target - sum(floors.values())
    order = sorted(raw.keys(), key=lambda k: (raw[k] - floors[k], k), reverse=True)
    for k in order[:rem]:
        floors[k] += 1
    return floors


def _subsample_category(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[str(r.get("source", "unknown"))].append(r)
    for src in by_src:
        rng.shuffle(by_src[src])
    quotas = _largest_remainder({s: len(v) for s, v in by_src.items()}, n)
    picked: list[dict] = []
    for src, q in quotas.items():
        picked.extend(by_src[src][:q])
    # top-up if rounding left us short
    if len(picked) < n:
        leftover = [r for r in rows if r not in picked]
        rng.shuffle(leftover)
        picked.extend(leftover[: n - len(picked)])
    return picked[:n]


def _try_resolve_stackselect(n_needed: int = 249) -> list[dict]:
    """Best-effort AdaLEval stackselect prompts for mixed tier."""
    url = "http://opencompass.openxlab.space/utils/AdaLEval/stackselect_1k.json"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"WARN stackselect fetch failed: {e}")
        return []
    # AdaLEval format varies; accept list of strings or dicts with prompt/question.
    prompts = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                prompts.append(item.strip())
            elif isinstance(item, dict):
                for k in ("prompt", "question", "input", "query"):
                    if item.get(k):
                        prompts.append(str(item[k]).strip())
                        break
    out = []
    for i, p in enumerate(prompts[:n_needed]):
        out.append(
            {
                "question_id": f"stackselect_{i:04d}",
                "category": "mixed",
                "source": url,
                "messages": [{"role": "user", "content": p}],
            }
        )
    return out


def prepare_speed747() -> Path:
    ds = load_dataset("nvidia/SPEED-Bench", "throughput_1k", split="test")
    resolved: list[dict] = []
    dropped_hle = 0
    unresolved = 0
    for r in ds:
        turns = r["turns"]
        src = r.get("source") or ""
        is_hle = "cais/hle" in src
        text = "\n".join(str(t) for t in turns) if isinstance(turns, list) else str(turns)
        if PH in text:
            if is_hle:
                dropped_hle += 1
                continue
            unresolved += 1
            continue
        msgs = [{"role": "user", "content": str(t)} for t in turns if str(t).strip()]
        if not msgs:
            continue
        resolved.append(
            {
                "question_id": r["question_id"],
                "category": r["category"],
                "source": src,
                "messages": msgs,
            }
        )

    # Fill mixed from stackselect if we dropped all mixed placeholders.
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in resolved:
        by_cat[row["category"]].append(row)

    if len(by_cat.get("mixed", [])) < 249:
        extra = _try_resolve_stackselect(249)
        # Dedup by content
        seen = {m["messages"][0]["content"] for m in by_cat.get("mixed", [])}
        for row in extra:
            c = row["messages"][0]["content"]
            if c in seen:
                continue
            by_cat["mixed"].append(row)
            seen.add(c)

    rng = random.Random(SEED)
    low = _subsample_category(by_cat.get("low_entropy", []), 249, rng)
    high = _subsample_category(by_cat.get("high_entropy", []), 249, rng)
    mixed_pool = by_cat.get("mixed", [])
    if len(mixed_pool) >= 249:
        mixed = _subsample_category(mixed_pool, 249, rng)
    else:
        mixed = list(mixed_pool)
        print(f"WARN mixed only {len(mixed)} < 249")

    final = low + high + mixed
    rng.shuffle(final)

    # Emit multi_turn JSONL for aiperf --custom-dataset-type multi_turn.
    # First user content only (throughput-style single request per prompt).
    out_rows = []
    for row in final:
        content = row["messages"][0]["content"]
        out_rows.append(
            {
                "session_id": row["question_id"],
                "turns": [{"text": content}],
            }
        )
    out = OUT / "speed_bench_throughput_1k_balanced_747.jsonl"
    write_jsonl(out, out_rows)
    print(
        f"speed747: n={len(out_rows)} dropped_hle={dropped_hle} "
        f"unresolved_other={unresolved} "
        f"cats={Counter(r['category'] for r in final)}"
    )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = prepare_spec_bench()
    speed = prepare_speed747()
    manifest = {
        "seed": SEED,
        "spec_bench": {
            "path": str(spec),
            "n": sum(1 for _ in spec.open()),
            "sha256": sha256_file(spec),
        },
        "speed747": {
            "path": str(speed),
            "n": sum(1 for _ in speed.open()),
            "sha256": sha256_file(speed),
            "note": (
                "Local derivative: drop unresolved HLE mixed (263), "
                "subsample low/high to 249 with seed 20260911 "
                "(source-stratified largest-remainder); mixed from "
                "resolved/AdaLEval stackselect."
            ),
        },
    }
    man = OUT / "MANIFEST.json"
    man.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
