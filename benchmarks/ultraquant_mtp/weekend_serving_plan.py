#!/usr/bin/env python3
"""Expand the weekend serving manifest into a resumable cell plan.

A variant runs only when it is enabled and its ``env`` object is non-empty.
``env_defaults`` is documentation and is never exported. ``WEEKEND_VARIANT_NAME``
plus ``WEEKEND_VARIANT_ENV`` or ``WEEKEND_VARIANT_ENV_FILE`` supplies an env
implementation for one named variant without editing the manifest.

This module does not start a server.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

NAME_RE_DESC = "lowercase letter, then lowercase letters, digits, or underscores"
ENV_NAME_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
DENIED_ENV = {
    "PATH",
    "PYTHONPATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "BASH_ENV",
    "ENV",
    "SHELLOPTS",
    "BASHOPTS",
    "IFS",
    "HOME",
    "USER",
    "LOGNAME",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
}


def workspace_from(manifest: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("WEEKEND_WORKSPACE")
    if env:
        return Path(env).resolve()
    # benchmarks/ultraquant_mtp -> vllm -> workspace
    return manifest.resolve().parents[3]


def resolve_under(workspace: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return workspace / path


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if int(data.get("schema_version", 0)) != 1:
        raise SystemExit(f"{path}: schema_version must be 1")
    names = [item["name"] for item in data["variants"]]
    if len(names) != len(set(names)):
        raise SystemExit(f"{path}: duplicate variant name")
    for name in names:
        if not valid_variant_name(name):
            raise SystemExit(f"{path}: invalid variant name {name!r} ({NAME_RE_DESC})")
    phases = data["phases"]
    seen = [item["name"] for item in phases]
    if seen != ["screen", "confirm"]:
        raise SystemExit(f"{path}: phases must be screen then confirm")
    by_name = {item["name"]: int(item["seconds"]) for item in phases}
    if by_name["screen"] != 180 or by_name["confirm"] != 300:
        raise SystemExit(f"{path}: screen must be 180s and confirm must be 300s")
    conc = [int(item) for item in data["concurrencies"]]
    if conc != [8, 16, 32]:
        raise SystemExit(f"{path}: concurrencies must be 8, 16, 32")
    if int(data["tp"]) != 1 or int(data["mtp"]) != 5 or str(data["hip"]) != "0":
        raise SystemExit(f"{path}: this queue is fixed to TP 1, MTP 5, HIP 0")
    return data


def valid_variant_name(name: str) -> bool:
    if not name or name[0] not in "abcdefghijklmnopqrstuvwxyz":
        return False
    return all(ch in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in name)


def parse_env_assignments(text: str) -> dict[str, str]:
    """Parse comma-separated or newline-separated KEY=VALUE assignments."""
    raw = text.strip()
    if not raw:
        return {}
    chunks: list[str] = []
    if "\n" in raw:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chunks.extend(part.strip() for part in line.split(","))
    else:
        chunks = [part.strip() for part in raw.split(",")]
    env: dict[str, str] = {}
    for chunk in chunks:
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"env assignment missing '=': {chunk}")
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        check_env_key(key)
        env[key] = value
    return env


def check_env_key(key: str) -> None:
    if not key or key[0].isdigit() or any(ch not in ENV_NAME_OK for ch in key):
        raise SystemExit(f"invalid env name: {key}")
    if key in DENIED_ENV:
        raise SystemExit(f"env name is not allowed on a variant: {key}")


def check_env_map(env: dict, where: str) -> dict[str, str]:
    if not isinstance(env, dict):
        raise SystemExit(f"{where}: env must be an object")
    clean: dict[str, str] = {}
    for key, value in env.items():
        check_env_key(str(key))
        clean[str(key)] = str(value)
    return clean


def overlay_from_environ() -> tuple[str, dict[str, str]] | None:
    name = os.environ.get("WEEKEND_VARIANT_NAME", "").strip()
    blob = os.environ.get("WEEKEND_VARIANT_ENV", "")
    file_name = os.environ.get("WEEKEND_VARIANT_ENV_FILE", "").strip()
    if not name and not blob.strip() and not file_name:
        return None
    if not name:
        raise SystemExit("WEEKEND_VARIANT_ENV is set without WEEKEND_VARIANT_NAME")
    if not valid_variant_name(name):
        raise SystemExit(f"invalid WEEKEND_VARIANT_NAME {name!r}")
    env: dict[str, str] = {}
    if file_name:
        path = Path(file_name)
        if not path.is_file():
            raise SystemExit(f"WEEKEND_VARIANT_ENV_FILE not found: {path}")
        env.update(parse_env_assignments(path.read_text()))
    if blob.strip():
        env.update(parse_env_assignments(blob))
    if not env:
        raise SystemExit(
            f"variant {name} has no env implementation; set WEEKEND_VARIANT_ENV "
            "or WEEKEND_VARIANT_ENV_FILE"
        )
    return name, env


def block_reason(enabled: bool, env: dict[str, str], overlay_applied: bool) -> str:
    if overlay_applied:
        return ""
    missing_env = not env
    if not enabled and missing_env:
        return "disabled; env implementation missing"
    if not enabled:
        return "disabled"
    if missing_env:
        return "env implementation missing"
    return ""


def build_plan(manifest: dict, workspace: Path) -> dict:
    overlay = overlay_from_environ()
    overlay_name = overlay[0] if overlay else ""
    overlay_env = overlay[1] if overlay else {}
    known = {item["name"] for item in manifest["variants"]}
    variants: list[dict] = []
    for item in manifest["variants"]:
        env = check_env_map(item.get("env") or {}, item["name"])
        enabled = bool(item.get("enabled"))
        applied = False
        if item["name"] == overlay_name:
            env = dict(overlay_env)
            enabled = True
            applied = True
        reason = block_reason(enabled, env, applied)
        variants.append(
            {
                "name": item["name"],
                "enabled": enabled,
                "runnable": reason == "",
                "block": reason,
                "env": env if reason == "" else {},
                "env_defaults": {
                    str(k): str(v) for k, v in (item.get("env_defaults") or {}).items()
                },
                "notes": item.get("notes", ""),
                "overlay": applied,
            }
        )
    if overlay_name and overlay_name not in known:
        variants.append(
            {
                "name": overlay_name,
                "enabled": True,
                "runnable": True,
                "block": "",
                "env": dict(overlay_env),
                "env_defaults": {},
                "notes": "Supplied by WEEKEND_VARIANT_NAME for this invocation.",
                "overlay": True,
            }
        )

    cells: list[dict] = []
    for variant in variants:
        if not variant["runnable"]:
            continue
        for conc in manifest["concurrencies"]:
            for phase in manifest["phases"]:
                phase_name = phase["name"]
                cell_id = (
                    f"qwen38_tp1_mtp5_{variant['name']}_c{int(conc)}_{phase_name}"
                )
                cells.append(
                    {
                        "cell_id": cell_id,
                        "variant": variant["name"],
                        "concurrency": int(conc),
                        "phase": phase_name,
                        "seconds": int(phase["seconds"]),
                        "kv_cache_dtype": manifest["kv_cache_dtype"],
                        "needs_screen": phase_name == "confirm",
                        "env": variant["env"],
                    }
                )
    gate = resolve_under(workspace, manifest["gate_marker"])
    output_root = resolve_under(workspace, manifest["output_root"])
    return {
        "model": manifest["model"],
        "tp": int(manifest["tp"]),
        "mtp": int(manifest["mtp"]),
        "hip": str(manifest["hip"]),
        "kv_cache_dtype": manifest["kv_cache_dtype"],
        "workload": manifest["workload"],
        "dataset": manifest["dataset"],
        "gate_marker": str(gate),
        "gate_present": gate.is_file(),
        "output_root": str(output_root),
        "variants": variants,
        "cells": cells,
    }


def write_plan(plan: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    env_dir = dest / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (dest / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    common = [
        f"MODEL={shlex.quote(plan['model'])}",
        f"TP={int(plan['tp'])}",
        f"MTP={int(plan['mtp'])}",
        f"HIP_DEVICE={shlex.quote(str(plan['hip']))}",
        f"KV_CACHE_DTYPE={shlex.quote(plan['kv_cache_dtype'])}",
        f"OUT_ROOT={shlex.quote(plan['output_root'])}",
        f"GATE_MARKER={shlex.quote(plan['gate_marker'])}",
    ]
    (dest / "common.env").write_text("\n".join(common) + "\n")
    rows = []
    for cell in plan["cells"]:
        env_path = env_dir / f"{cell['cell_id']}.env"
        lines = [
            f"{key}={shlex.quote(value)}" for key, value in sorted(cell["env"].items())
        ]
        env_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append(
            "\t".join(
                [
                    cell["cell_id"],
                    cell["variant"],
                    str(cell["concurrency"]),
                    cell["phase"],
                    str(cell["seconds"]),
                    cell["kv_cache_dtype"],
                    "1" if cell["needs_screen"] else "0",
                    str(env_path),
                ]
            )
        )
    (dest / "cells.tsv").write_text(("\n".join(rows) + "\n") if rows else "")


def format_summary(plan: dict) -> str:
    runnable = [item["name"] for item in plan["variants"] if item["runnable"]]
    lines = [
        f"gate_marker={plan['gate_marker']} present={str(plan['gate_present']).lower()}",
        f"output_root={plan['output_root']}",
        f"runnable_variants={','.join(runnable) if runnable else '(none)'}",
        f"cells={len(plan['cells'])}",
    ]
    for item in plan["variants"]:
        state = "runnable" if item["runnable"] else item["block"]
        keys = ",".join(item["env"]) if item["env"] else "-"
        lines.append(f"variant {item['name']}: {state} env={keys}")
    for cell in plan["cells"]:
        lines.append(
            f"cell {cell['cell_id']} conc={cell['concurrency']} "
            f"phase={cell['phase']} seconds={cell['seconds']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("weekend_serving_manifest.json"),
    )
    parser.add_argument("--workspace", default=None)
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Write plan.json, common.env, cells.tsv, and per-cell env files.",
    )
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    workspace = workspace_from(args.manifest, args.workspace)
    plan = build_plan(manifest, workspace)
    if args.write:
        write_plan(plan, args.write)
    print(format_summary(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
