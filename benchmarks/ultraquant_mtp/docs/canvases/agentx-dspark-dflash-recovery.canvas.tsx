import {
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
} from "cursor/canvas";

/**
 * Phase-2 AgentX DSpark/DFlash recovery (geom64 + mnbt8k + SWA win_table + adaptive).
 * Source: vllm/out/agentx_dspark_dflash_recovery
 */

type Row = {
  arm: string;
  conc: number;
  tok_s: number | null;
  accept: number | null;
  prefix_hit: number | null;
  vs_nospec: number | null;
};

const ROWS: Row[] = [];
const SUITE_STATUS: "pending" | "running" | "done" = "pending";
const REFRESHED = "2026-10-29 — suite queued after Phase-1 lit-match";

function fmt(n: number | null, d = 1): string {
  if (n == null) return "—";
  return n.toFixed(d);
}

export default function AgentxDsparkDflashRecovery() {
  const done = ROWS.filter((r) => r.tok_s != null).length;
  return (
    <Stack gap={24}>
      <Stack gap={6}>
        <H1>AgentX recovery — DSpark / DFlash geom64</H1>
        <Text tone="secondary">
          Phase 2: <Code>block_size=64</Code>, <Code>mnbt=8192</Code>, retention
          49920/50176, FP8 draft kv8, DFlash{" "}
          <Code>ROCM_AITER_FA_DFLASH_SWA</Code> +{" "}
          <Code>VLLM_DFLASH_DRAFT_ATTN_EXPERIMENT=win_table</Code>, adaptive A/B.
          Compare to baseline nospec in{" "}
          <Code>agentx_baseline_qwen38</Code>. Refreshed {REFRESHED}.
        </Text>
        <Row gap={8} wrap>
          <Pill tone="info">suite {SUITE_STATUS}</Pill>
          <Pill>{done} cells with tok/s</Pill>
        </Row>
      </Stack>

      <Callout tone="info" title="C32 gates">
        ≥1.15× baseline nospec · prefix hit ≥70% · accept length ≥2.5 · ITL not
        10× worse than MTP.
      </Callout>

      <H2>Config deltas vs broken AgentX baseline</H2>
      <Table
        headers={["Knob", "Baseline (broken)", "Recovery"]}
        rows={[
          ["block_size", "default (FA→16 base)", "64"],
          ["mnbt", "2048", "8192"],
          ["retention", "49920 / 50176", "same (verify LCM after geom)"],
          ["DFlash draft attn", "TRITON / FA_DFLASH_SWA", "FA_DFLASH_SWA + win_table"],
          ["FP8 draft KV", "fp8", "fp8 (keep page match)"],
        ]}
        striped
      />

      <H2>Results</H2>
      {ROWS.length === 0 ? (
        <Text tone="secondary">
          Waiting for Phase 1 lit-match GPU, then{" "}
          <Code>agentx_dspark_dflash_recovery/run_suite.sh</Code>.
        </Text>
      ) : (
        <Table
          headers={["Arm", "C", "tok/s", "vs nospec", "accept", "prefix hit"]}
          columnAlign={["left", "right", "right", "right", "right", "right"]}
          rows={ROWS.map((r) => [
            r.arm,
            String(r.conc),
            fmt(r.tok_s),
            r.vs_nospec != null ? `${r.vs_nospec.toFixed(2)}×` : "—",
            fmt(r.accept, 2),
            r.prefix_hit != null ? `${(100 * r.prefix_hit).toFixed(0)}%` : "—",
          ])}
          striped
        />
      )}

      <Divider />
      <Text size="small" tone="tertiary">
        Suite:{" "}
        <Code>vllm/benchmarks/ultraquant_mtp/agentx_dspark_dflash_recovery/</Code>
      </Text>
    </Stack>
  );
}
