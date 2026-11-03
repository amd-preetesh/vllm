import {
  Code,
  H1,
  Stack,
  Table,
  Text,
} from "cursor/canvas";

/** AgentX SA seed42 · output tok/s · 2026-10-26 */
const ROWS = [
  ["8", "21.5", "67.3", "67.3", "67.3"],
  ["16", "44.5", "55.9", "132.3", "101.7"],
  ["20", "76.2", "79.3", "155.7", "129.4"],
  ["24", "65.8", "68.6", "149.4", "101.6"],
  ["30", "94.0", "112.5", "163.4", "103.3"],
  ["32", "105.2", "105.2", "185.3", "115.0"],
];

export default function AgentxToksConcTable() {
  return (
    <Stack gap={16}>
      <Stack gap={6}>
        <H1>AgentX tok/s by concurrency</H1>
        <Text tone="secondary">
          Output throughput (tok/s) · 6 concs × 4 arms (FP8 FA new baseline vs
          best UQ per method). Source:{" "}
          <Code>*_fp8_fair_fa</Code>, <Code>*_uq_opt</Code>,{" "}
          <Code>*_uq_dflash_swa</Code> · SA seed42.
        </Text>
      </Stack>

      <Table
        headers={[
          "Conc",
          "DSpark FP8 FA",
          "DSpark UQ opt",
          "DFlash FP8 FA",
          "DFlash UQ SWA",
        ]}
        columnAlign={["right", "right", "right", "right", "right"]}
        rows={ROWS}
        striped
      />

      <Text size="small" tone="tertiary">
        FP8 FA = tgt ROCM_AITER_FA · draft TRITON · K=7. DSpark UQ opt = draft
        FA_UQ_DENSE. DFlash UQ SWA = draft FA_DFLASH_SWA.
      </Text>
    </Stack>
  );
}
