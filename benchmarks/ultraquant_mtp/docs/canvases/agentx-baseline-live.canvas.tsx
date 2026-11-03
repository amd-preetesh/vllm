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
  UsageBar,
} from "cursor/canvas";

/** Auto-refreshed by agentx_baseline_matrix/refresh_progress.py */
export default function AgentxBaselineLive() {
  return (
    <Stack gap={24}>
      <Stack gap={6}>
        <H1>AgentX baseline matrix — Qwen3.8-27B</H1>
        <Text tone="secondary">
          TP=1 · C∈{8,16,24,32,36} · HF thinking card (T=1.0, top_p=0.95, top_k=20,
          presence=0, effort=xhigh). FP8 uses <Code>ROCM_AITER_FA</Code>; UQ uses
          UNIFIED / SPEC_COMPAT + UQ draft kernels. Refreshed 2026-10-29 11:07 PDT.
        </Text>
        <Row gap={8} wrap>
          <Pill tone="success">55/55 cells</Pill>
          <Pill tone="info">0 running</Pill>
          <Pill tone="warning">0 fail</Pill>
          <Pill>0 pending</Pill>
        </Row>
      </Stack>

      <Row gap={16} wrap>
        <Stat value="55 / 55" label="AgentX cells done" tone="success" />
        <Stat value="0" label="Measuring now" tone="info" />
        <Stat value="0" label="Failed" tone="warning" />
        <Stat value="2" label="GPU lanes (FP8 | UQ)" />
      </Row>

      <UsageBar
        segments={[
          { label: "done", value: 55, tone: "success" },
          { label: "running", value: 0, tone: "info" },
          { label: "fail", value: 0, tone: "warning" },
          { label: "pending", value: 0 },
        ]}
      />

      <H2>Throughput by arm (tok/s)</H2>
      <Table
        headers={[
          "Arm",
          "Lane",
          "Progress",
          "Status",
          "C8",
          "C16",
          "C24",
          "C32",
          "C36",
        ]}
        columnAlign={[
          "left",
          "left",
          "right",
          "left",
          "right",
          "right",
          "right",
          "right",
          "right",
        ]}
        rows={[
          ["FP8 + no spec", "FP8", "5/5", "done", "69.3", "55.9", "68.6", "118.4", "160.9"],
          ["UQ + no spec", "UQ", "5/5", "done", "67.3", "55.9", "101.6", "132.7", "160.9"],
          ["FP8 + MTP", "FP8", "5/5", "done", "67.3", "133.1", "192.3", "241.9", "197.2"],
          ["UQ + MTP (prod FlyDSL)", "UQ", "5/5", "done", "67.3", "132.3", "147.4", "118.0", "187.9"],
          ["UQ* + MTP (PDV1.1 FlyDSL)", "UQ", "5/5", "done", "155.9", "132.3", "175.2", "227.1", "171.1"],
          ["FP8 + DFlash2", "FP8", "5/5", "done", "67.3", "136.3", "149.4", "184.0", "151.1"],
          ["UQ + DFlash2", "UQ", "5/5", "done", "67.3", "101.7", "101.6", "115.5", "141.7"],
          ["FP8 + DSpark", "FP8", "5/5", "done", "21.5", "44.5", "65.8", "96.6", "66.7"],
          ["UQ + DSpark", "UQ", "5/5", "done", "69.3", "55.9", "70.8", "105.2", "93.0"],
          ["FP8 + DSpark adaptive", "FP8", "5/5", "done", "21.5", "44.5", "65.9", "105.2", "58.1"],
          ["UQ + DSpark adaptive", "UQ", "5/5", "done", "67.3", "55.9", "65.9", "115.2", "104.4"]
        ]}
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
}
