import {
  Callout,
  Code,
  Divider,
  H1,
  H2,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
  useState,
  type Color,
} from "cursor/canvas";

type ArmId =
  | "fp8_nospec"
  | "uq_nospec"
  | "fp8_mtp"
  | "uq_mtp_prod"
  | "uq_mtp_pdv"
  | "fp8_dflash"
  | "uq_dflash"
  | "fp8_dspark"
  | "uq_dspark"
  | "fp8_dspark_adaptive"
  | "uq_dspark_adaptive";

type Point = {
  arm: ArmId;
  label: string;
  lane: "fp8" | "uq";
  conc: number;
  toks: number;
  itlAvg: number;
  ttft: number;
};

/** AgentX SA seed42 · C∈{8,16,24,32,36} · HF thinking · 300s · refreshed 2026-10-29 */
const POINTS: Point[] = [
  { arm: "fp8_nospec", label: "FP8 + no spec", lane: "fp8", conc: 8, toks: 69.3, itlAvg: 13.5, ttft: 4.42 },
  { arm: "fp8_nospec", label: "FP8 + no spec", lane: "fp8", conc: 16, toks: 55.9, itlAvg: 20.9, ttft: 2.52 },
  { arm: "fp8_nospec", label: "FP8 + no spec", lane: "fp8", conc: 24, toks: 68.6, itlAvg: 16.1, ttft: 2.02 },
  { arm: "fp8_nospec", label: "FP8 + no spec", lane: "fp8", conc: 32, toks: 118.4, itlAvg: 26.4, ttft: 2.22 },
  { arm: "fp8_nospec", label: "FP8 + no spec", lane: "fp8", conc: 36, toks: 160.9, itlAvg: 38.9, ttft: 1.7 },
  { arm: "uq_nospec", label: "UQ + no spec", lane: "uq", conc: 8, toks: 67.3, itlAvg: 12.3, ttft: 2.41 },
  { arm: "uq_nospec", label: "UQ + no spec", lane: "uq", conc: 16, toks: 55.9, itlAvg: 17.7, ttft: 2.3 },
  { arm: "uq_nospec", label: "UQ + no spec", lane: "uq", conc: 24, toks: 101.6, itlAvg: 15.6, ttft: 1.86 },
  { arm: "uq_nospec", label: "UQ + no spec", lane: "uq", conc: 32, toks: 132.7, itlAvg: 25.9, ttft: 1.46 },
  { arm: "uq_nospec", label: "UQ + no spec", lane: "uq", conc: 36, toks: 160.9, itlAvg: 49.7, ttft: 2.53 },
  { arm: "fp8_mtp", label: "FP8 + MTP", lane: "fp8", conc: 8, toks: 67.3, itlAvg: 8.5, ttft: 2.99 },
  { arm: "fp8_mtp", label: "FP8 + MTP", lane: "fp8", conc: 16, toks: 133.1, itlAvg: 11.8, ttft: 2.22 },
  { arm: "fp8_mtp", label: "FP8 + MTP", lane: "fp8", conc: 24, toks: 192.3, itlAvg: 11.8, ttft: 2.59 },
  { arm: "fp8_mtp", label: "FP8 + MTP", lane: "fp8", conc: 32, toks: 241.9, itlAvg: 19.4, ttft: 2.39 },
  { arm: "fp8_mtp", label: "FP8 + MTP", lane: "fp8", conc: 36, toks: 197.2, itlAvg: 37.8, ttft: 2.66 },
  { arm: "uq_mtp_prod", label: "UQ + MTP (prod FlyDSL)", lane: "uq", conc: 8, toks: 67.3, itlAvg: 9.0, ttft: 3.42 },
  { arm: "uq_mtp_prod", label: "UQ + MTP (prod FlyDSL)", lane: "uq", conc: 16, toks: 132.3, itlAvg: 14.2, ttft: 2.71 },
  { arm: "uq_mtp_prod", label: "UQ + MTP (prod FlyDSL)", lane: "uq", conc: 24, toks: 147.4, itlAvg: 16.0, ttft: 3.18 },
  { arm: "uq_mtp_prod", label: "UQ + MTP (prod FlyDSL)", lane: "uq", conc: 32, toks: 118.0, itlAvg: 33.3, ttft: 3.05 },
  { arm: "uq_mtp_prod", label: "UQ + MTP (prod FlyDSL)", lane: "uq", conc: 36, toks: 187.9, itlAvg: 51.0, ttft: 4.26 },
  { arm: "uq_mtp_pdv", label: "UQ* + MTP (PDV1.1)", lane: "uq", conc: 8, toks: 155.9, itlAvg: 7.1, ttft: 3.57 },
  { arm: "uq_mtp_pdv", label: "UQ* + MTP (PDV1.1)", lane: "uq", conc: 16, toks: 132.3, itlAvg: 10.0, ttft: 2.72 },
  { arm: "uq_mtp_pdv", label: "UQ* + MTP (PDV1.1)", lane: "uq", conc: 24, toks: 175.2, itlAvg: 12.6, ttft: 2.9 },
  { arm: "uq_mtp_pdv", label: "UQ* + MTP (PDV1.1)", lane: "uq", conc: 32, toks: 227.1, itlAvg: 22.1, ttft: 3.27 },
  { arm: "uq_mtp_pdv", label: "UQ* + MTP (PDV1.1)", lane: "uq", conc: 36, toks: 171.1, itlAvg: 43.0, ttft: 3.95 },
  { arm: "fp8_dflash", label: "FP8 + DFlash2", lane: "fp8", conc: 8, toks: 67.3, itlAvg: 8.3, ttft: 2.8 },
  { arm: "fp8_dflash", label: "FP8 + DFlash2", lane: "fp8", conc: 16, toks: 136.3, itlAvg: 10.7, ttft: 2.35 },
  { arm: "fp8_dflash", label: "FP8 + DFlash2", lane: "fp8", conc: 24, toks: 149.4, itlAvg: 12.2, ttft: 2.75 },
  { arm: "fp8_dflash", label: "FP8 + DFlash2", lane: "fp8", conc: 32, toks: 184.0, itlAvg: 17.3, ttft: 2.32 },
  { arm: "fp8_dflash", label: "FP8 + DFlash2", lane: "fp8", conc: 36, toks: 151.1, itlAvg: 41.6, ttft: 5.71 },
  { arm: "uq_dflash", label: "UQ + DFlash2", lane: "uq", conc: 8, toks: 67.3, itlAvg: 9.3, ttft: 5.11 },
  { arm: "uq_dflash", label: "UQ + DFlash2", lane: "uq", conc: 16, toks: 101.7, itlAvg: 13.7, ttft: 3.89 },
  { arm: "uq_dflash", label: "UQ + DFlash2", lane: "uq", conc: 24, toks: 101.6, itlAvg: 20.2, ttft: 4.39 },
  { arm: "uq_dflash", label: "UQ + DFlash2", lane: "uq", conc: 32, toks: 115.5, itlAvg: 29.4, ttft: 4.31 },
  { arm: "uq_dflash", label: "UQ + DFlash2", lane: "uq", conc: 36, toks: 141.7, itlAvg: 60.3, ttft: 6.95 },
  { arm: "fp8_dspark", label: "FP8 + DSpark", lane: "fp8", conc: 8, toks: 21.5, itlAvg: 21.1, ttft: 3.46 },
  { arm: "fp8_dspark", label: "FP8 + DSpark", lane: "fp8", conc: 16, toks: 44.5, itlAvg: 22.4, ttft: 2.63 },
  { arm: "fp8_dspark", label: "FP8 + DSpark", lane: "fp8", conc: 24, toks: 65.8, itlAvg: 25.6, ttft: 2.9 },
  { arm: "fp8_dspark", label: "FP8 + DSpark", lane: "fp8", conc: 32, toks: 96.6, itlAvg: 35.5, ttft: 2.28 },
  { arm: "fp8_dspark", label: "FP8 + DSpark", lane: "fp8", conc: 36, toks: 66.7, itlAvg: 84.0, ttft: 25.17 },
  { arm: "uq_dspark", label: "UQ + DSpark", lane: "uq", conc: 8, toks: 69.3, itlAvg: 13.7, ttft: 3.46 },
  { arm: "uq_dspark", label: "UQ + DSpark", lane: "uq", conc: 16, toks: 55.9, itlAvg: 17.7, ttft: 2.39 },
  { arm: "uq_dspark", label: "UQ + DSpark", lane: "uq", conc: 24, toks: 70.8, itlAvg: 19.3, ttft: 2.95 },
  { arm: "uq_dspark", label: "UQ + DSpark", lane: "uq", conc: 32, toks: 105.2, itlAvg: 33.8, ttft: 2.46 },
  { arm: "uq_dspark", label: "UQ + DSpark", lane: "uq", conc: 36, toks: 93.0, itlAvg: 77.5, ttft: 9.12 },
  { arm: "fp8_dspark_adaptive", label: "FP8 + DSpark adaptive", lane: "fp8", conc: 8, toks: 21.5, itlAvg: 25.6, ttft: 3.58 },
  { arm: "fp8_dspark_adaptive", label: "FP8 + DSpark adaptive", lane: "fp8", conc: 16, toks: 44.5, itlAvg: 21.7, ttft: 2.52 },
  { arm: "fp8_dspark_adaptive", label: "FP8 + DSpark adaptive", lane: "fp8", conc: 24, toks: 65.9, itlAvg: 28.9, ttft: 2.9 },
  { arm: "fp8_dspark_adaptive", label: "FP8 + DSpark adaptive", lane: "fp8", conc: 32, toks: 105.2, itlAvg: 35.6, ttft: 2.0 },
  { arm: "fp8_dspark_adaptive", label: "FP8 + DSpark adaptive", lane: "fp8", conc: 36, toks: 58.1, itlAvg: 86.9, ttft: 22.94 },
  { arm: "uq_dspark_adaptive", label: "UQ + DSpark adaptive", lane: "uq", conc: 8, toks: 67.3, itlAvg: 14.9, ttft: 3.47 },
  { arm: "uq_dspark_adaptive", label: "UQ + DSpark adaptive", lane: "uq", conc: 16, toks: 55.9, itlAvg: 15.4, ttft: 2.58 },
  { arm: "uq_dspark_adaptive", label: "UQ + DSpark adaptive", lane: "uq", conc: 24, toks: 65.9, itlAvg: 33.7, ttft: 3.31 },
  { arm: "uq_dspark_adaptive", label: "UQ + DSpark adaptive", lane: "uq", conc: 32, toks: 115.2, itlAvg: 36.8, ttft: 2.41 },
  { arm: "uq_dspark_adaptive", label: "UQ + DSpark adaptive", lane: "uq", conc: 36, toks: 104.4, itlAvg: 78.0, ttft: 8.31 },
];

const ARM_ORDER: ArmId[] = [
  "fp8_nospec",
  "uq_nospec",
  "fp8_mtp",
  "uq_mtp_prod",
  "uq_mtp_pdv",
  "fp8_dflash",
  "uq_dflash",
  "fp8_dspark",
  "uq_dspark",
  "fp8_dspark_adaptive",
  "uq_dspark_adaptive",
];

const ARM_LABEL: Record<ArmId, string> = {
  fp8_nospec: "FP8 + no spec",
  uq_nospec: "UQ + no spec",
  fp8_mtp: "FP8 + MTP",
  uq_mtp_prod: "UQ + MTP (prod)",
  uq_mtp_pdv: "UQ* + MTP (PDV1.1)",
  fp8_dflash: "FP8 + DFlash2",
  uq_dflash: "UQ + DFlash2",
  fp8_dspark: "FP8 + DSpark",
  uq_dspark: "UQ + DSpark",
  fp8_dspark_adaptive: "FP8 + DSpark adaptive",
  uq_dspark_adaptive: "UQ + DSpark adaptive",
};

/** One unique hue per arm — category tokens + accent/link for the last two. */
const ARM_COLOR_KEY: Record<ArmId, Color | "accent" | "link"> = {
  fp8_nospec: "blue",
  uq_nospec: "cyan",
  fp8_mtp: "purple",
  uq_mtp_prod: "green",
  uq_mtp_pdv: "yellow",
  fp8_dflash: "pink",
  uq_dflash: "orange",
  fp8_dspark: "red",
  uq_dspark: "gray",
  fp8_dspark_adaptive: "accent",
  uq_dspark_adaptive: "link",
};

function armColor(
  theme: ReturnType<typeof useHostTheme>,
  arm: ArmId,
): string {
  const key = ARM_COLOR_KEY[arm];
  if (key === "accent") return theme.accent.primary;
  if (key === "link") return theme.text.link;
  return theme.category[key];
}

function dominates(a: Point, b: Point): boolean {
  const betterOrEq = a.itlAvg <= b.itlAvg && a.toks >= b.toks;
  const strict = a.itlAvg < b.itlAvg || a.toks > b.toks;
  return betterOrEq && strict;
}

function frontier(points: Point[]): Point[] {
  return points.filter((p) => !points.some((q) => q !== p && dominates(q, p)));
}

function ticks(min: number, max: number, count: number): number[] {
  const rough = (max - min) / Math.max(1, count - 1);
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(rough, 1e-9))));
  const err = rough / pow;
  const step =
    err >= 7.5 ? 10 * pow : err >= 3.5 ? 5 * pow : err >= 1.5 ? 2 * pow : pow;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + step * 0.001; v += step) {
    out.push(Math.round(v * 100) / 100);
  }
  return out;
}

function fmt(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  return v.toFixed(1);
}

function peakOf(arm: ArmId): Point {
  return POINTS.filter((p) => p.arm === arm).reduce((a, b) =>
    b.toks > a.toks ? b : a,
  );
}

function Marker({
  cx,
  cy,
  r,
  fill,
  stroke,
  lane,
}: {
  cx: number;
  cy: number;
  r: number;
  fill: string;
  stroke: string;
  lane: "fp8" | "uq";
}) {
  if (lane === "uq") {
    const s = r * 1.15;
    return (
      <rect
        x={cx - s}
        y={cy - s}
        width={s * 2}
        height={s * 2}
        transform={`rotate(45 ${cx} ${cy})`}
        fill={fill}
        stroke={stroke}
        strokeWidth={1.5}
      />
    );
  }
  return (
    <circle
      cx={cx}
      cy={cy}
      r={r}
      fill={fill}
      stroke={stroke}
      strokeWidth={1.5}
    />
  );
}

function ParetoPlot({
  points,
  hidden,
}: {
  points: Point[];
  hidden: Set<ArmId>;
}) {
  const theme = useHostTheme();
  const [hover, setHover] = useState<Point | null>(null);

  const visible = points.filter((p) => !hidden.has(p.arm));
  if (visible.length === 0) {
    return (
      <Text tone="secondary">Toggle at least one arm in the legend.</Text>
    );
  }

  const front = frontier(visible);
  const xs = visible.map((p) => p.itlAvg);
  const ys = visible.map((p) => p.toks);
  const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.1 || 1;
  const yPad = (Math.max(...ys) - Math.min(...ys)) * 0.14 || 1;
  const xMin = Math.max(0, Math.min(...xs) - xPad);
  const xMax = Math.max(...xs) + xPad;
  const yMin = Math.max(0, Math.min(...ys) - yPad);
  const yMax = Math.max(...ys) + yPad;

  const W = 920;
  const H = 480;
  const pad = { l: 72, r: 24, t: 18, b: 52 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const mapX = (x: number) => pad.l + ((x - xMin) / (xMax - xMin)) * iw;
  const mapY = (y: number) => pad.t + ((yMax - y) / (yMax - yMin)) * ih;

  const xTicks = ticks(xMin, xMax, 7);
  const yTicks = ticks(yMin, yMax, 6);
  const sortedFront = [...front].sort((a, b) => a.itlAvg - b.itlAvg);

  const tipW = 260;
  const tipH = 64;
  let tipX = 0;
  let tipY = 0;
  if (hover) {
    tipX = mapX(hover.itlAvg) + 12;
    tipY = mapY(hover.toks) - tipH - 8;
    if (tipX + tipW > W - 8) tipX = mapX(hover.itlAvg) - tipW - 12;
    if (tipY < 4) tipY = mapY(hover.toks) + 12;
  }

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      role="img"
      aria-label="Mean ITL vs output throughput by arm"
      onMouseLeave={() => setHover(null)}
    >
      {xTicks.map((t) => (
        <g key={`x${t}`}>
          <line
            x1={mapX(t)}
            x2={mapX(t)}
            y1={pad.t}
            y2={pad.t + ih}
            stroke={theme.stroke.secondary}
            strokeWidth={1}
            opacity={0.35}
          />
          <text
            x={mapX(t)}
            y={H - 18}
            textAnchor="middle"
            fill={theme.text.tertiary}
            fontSize={11}
          >
            {fmt(t)}
          </text>
        </g>
      ))}
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line
            x1={pad.l}
            x2={pad.l + iw}
            y1={mapY(t)}
            y2={mapY(t)}
            stroke={theme.stroke.secondary}
            strokeWidth={1}
            opacity={0.35}
          />
          <text
            x={pad.l - 10}
            y={mapY(t) + 4}
            textAnchor="end"
            fill={theme.text.tertiary}
            fontSize={11}
          >
            {fmt(t)}
          </text>
        </g>
      ))}

      <text
        x={W / 2}
        y={H - 4}
        textAnchor="middle"
        fill={theme.text.secondary}
        fontSize={12}
      >
        Mean inter-token latency (ms) → worse
      </text>
      <text
        x={16}
        y={H / 2}
        textAnchor="middle"
        fill={theme.text.secondary}
        fontSize={12}
        transform={`rotate(-90 16 ${H / 2})`}
      >
        Output throughput (tok/s) → better
      </text>

      {ARM_ORDER.filter((arm) => !hidden.has(arm)).map((arm) => {
        const pts = visible
          .filter((p) => p.arm === arm)
          .sort((a, b) => a.conc - b.conc);
        if (pts.length < 2) return null;
        const d = pts
          .map(
            (p, i) =>
              `${i === 0 ? "M" : "L"} ${mapX(p.itlAvg)} ${mapY(p.toks)}`,
          )
          .join(" ");
        return (
          <path
            key={arm}
            d={d}
            fill="none"
            stroke={armColor(theme, arm)}
            strokeWidth={2}
            opacity={0.45}
          />
        );
      })}

      {sortedFront.length >= 2 && (
        <path
          d={sortedFront
            .map(
              (p, i) =>
                `${i === 0 ? "M" : "L"} ${mapX(p.itlAvg)} ${mapY(p.toks)}`,
            )
            .join(" ")}
          fill="none"
          stroke={theme.text.tertiary}
          strokeWidth={1.5}
          strokeDasharray="5 4"
          opacity={0.75}
        />
      )}

      {visible.map((p) => {
        const cx = mapX(p.itlAvg);
        const cy = mapY(p.toks);
        const active = hover === p;
        return (
          <g
            key={`${p.arm}-${p.conc}`}
            onMouseEnter={() => setHover(p)}
            style={{ cursor: "pointer" }}
          >
            <Marker
              cx={cx}
              cy={cy}
              r={active ? 7 : 5}
              fill={armColor(theme, p.arm)}
              stroke={theme.bg.editor}
              lane={p.lane}
            />
            <text
              x={cx}
              y={cy - 11}
              textAnchor="middle"
              fill={theme.text.primary}
              fontSize={10}
              fontWeight={600}
            >
              {p.conc}
            </text>
          </g>
        );
      })}

      {hover && (
        <g>
          <rect
            x={tipX}
            y={tipY}
            width={tipW}
            height={tipH}
            rx={4}
            fill={theme.bg.elevated}
            stroke={theme.stroke.primary}
          />
          <text
            x={tipX + 10}
            y={tipY + 18}
            fill={theme.text.primary}
            fontSize={12}
            fontWeight={600}
          >
            {`C=${hover.conc} · ${hover.label}`}
          </text>
          <text x={tipX + 10} y={tipY + 36} fill={theme.text.secondary} fontSize={11}>
            {`${hover.toks} tok/s · ITL ${hover.itlAvg} ms`}
          </text>
          <text x={tipX + 10} y={tipY + 52} fill={theme.text.tertiary} fontSize={11}>
            {`TTFT ${hover.ttft}s`}
          </text>
        </g>
      )}
    </svg>
  );
}

function LegendSwatch({ arm }: { arm: ArmId }) {
  const theme = useHostTheme();
  const color = armColor(theme, arm);
  const lane = arm.startsWith("uq") ? "uq" : "fp8";
  if (lane === "uq") {
    return (
      <span
        style={{
          display: "inline-block",
          width: 10,
          height: 10,
          background: color,
          transform: "rotate(45deg)",
          flexShrink: 0,
        }}
      />
    );
  }
  return (
    <span
      style={{
        display: "inline-block",
        width: 10,
        height: 10,
        borderRadius: 999,
        background: color,
        flexShrink: 0,
      }}
    />
  );
}

export default function AgentxBaselinePareto() {
  const [hidden, setHidden] = useState<ArmId[]>([]);
  const hiddenSet = new Set(hidden);

  const toggle = (arm: ArmId) => {
    setHidden((prev) =>
      prev.includes(arm) ? prev.filter((a) => a !== arm) : [...prev, arm],
    );
  };

  const showAll = () => setHidden([]);
  const onlyLane = (lane: "fp8" | "uq") => {
    setHidden(ARM_ORDER.filter((a) => !a.startsWith(lane)));
  };

  const peaks = ARM_ORDER.map((arm) => peakOf(arm));
  const bestPeak = peaks.reduce((a, b) => (b.toks > a.toks ? b : a));
  const frontAll = frontier(POINTS).sort((a, b) => a.itlAvg - b.itlAvg);

  const concCats = ["8", "16", "24", "32", "36"];
  const lineSeries = ARM_ORDER.map((arm) => ({
    name: ARM_LABEL[arm],
    data: POINTS.filter((p) => p.arm === arm)
      .sort((a, b) => a.conc - b.conc)
      .map((p) => p.toks),
  }));

  return (
    <Stack gap={28}>
      <Stack gap={8}>
        <H1>AgentX baseline — ITL × throughput Pareto</H1>
        <Text tone="secondary">
          All 11 arms from{" "}
          <Code>agentx_baseline_qwen38</Code> · Qwen3.8-27B TP=1 · HF thinking
          card · C∈{"{8,16,24,32,36}"} · 300s SA seed42. X = mean ITL (ms), Y =
          output tok/s. Circles = FP8, diamonds = UQ. Dashed = joint frontier of
          visible arms.
        </Text>
        <Row gap={8} wrap>
          <Pill tone="success">Peak {bestPeak.toks} tok/s</Pill>
          <Pill tone="info">{bestPeak.label}</Pill>
          <Pill>@ C={bestPeak.conc}</Pill>
          <Pill tone="warning">{frontAll.length} frontier points</Pill>
        </Row>
      </Stack>

      <Row gap={16} wrap>
        <Stat
          value={`${bestPeak.toks}`}
          label={`Best peak · ${ARM_LABEL[bestPeak.arm]}`}
          tone="success"
        />
        <Stat
          value={`${peakOf("uq_mtp_pdv").toks}`}
          label="UQ* MTP PDV peak"
          tone="info"
        />
        <Stat
          value={`${peakOf("fp8_dflash").toks}`}
          label="FP8 DFlash2 peak"
        />
        <Stat
          value={`${peakOf("uq_dspark_adaptive").toks}`}
          label="UQ DSpark adaptive peak"
        />
      </Row>

      <Callout tone="info" title="Headline">
        FP8 MTP leads absolute throughput (241.9 tok/s @C32, ITL 19.4 ms). UQ*
        MTP PDV1.1 is close behind (227.1 @C32) and owns the low-ITL corner
        (155.9 tok/s @C8, ITL 7.1 ms). Dense drafts (DFlash/DSpark) sit below
        MTP; adaptive is roughly flat vs fixed-K DSpark.
      </Callout>

      <Stack gap={10}>
        <H2>Pareto — mean ITL vs tok/s</H2>
        <Row gap={8} wrap>
          <Pill
            tone={hidden.length === 0 ? "info" : "neutral"}
            active={hidden.length === 0}
            onClick={showAll}
          >
            Show all
          </Pill>
          <Pill onClick={() => onlyLane("fp8")}>FP8 only</Pill>
          <Pill onClick={() => onlyLane("uq")}>UQ only</Pill>
          <Text size="small" tone="tertiary">
            Click legend items to hide/show · hover dots for metrics
          </Text>
        </Row>

        <Row gap={10} wrap>
          {ARM_ORDER.map((arm) => {
            const on = !hiddenSet.has(arm);
            return (
              <button
                key={arm}
                type="button"
                onClick={() => toggle(arm)}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid transparent",
                  background: "transparent",
                  cursor: "pointer",
                  opacity: on ? 1 : 0.35,
                }}
              >
                <LegendSwatch arm={arm} />
                <Text size="small" tone={on ? "primary" : "tertiary"}>
                  {ARM_LABEL[arm]}
                </Text>
              </button>
            );
          })}
        </Row>

        <ParetoPlot points={POINTS} hidden={hiddenSet} />
      </Stack>

      <Divider />

      <Stack gap={10}>
        <H2>Throughput vs concurrency</H2>
        <Text tone="secondary">
          Same suite — output tok/s by concurrency ladder. Colors auto-assigned
          per series (unique legend entries).
        </Text>
        <LineChart
          categories={concCats}
          series={lineSeries}
          height={320}
          valueSuffix=" tok/s"
          beginAtZero
        />
      </Stack>

      <Stack gap={10}>
        <H2>Peak per arm</H2>
        <Table
          headers={["Arm", "Lane", "Peak C", "tok/s", "ITL (ms)", "TTFT (s)"]}
          columnAlign={["left", "left", "right", "right", "right", "right"]}
          rows={peaks
            .slice()
            .sort((a, b) => b.toks - a.toks)
            .map((p) => [
              ARM_LABEL[p.arm],
              p.lane.toUpperCase(),
              String(p.conc),
              p.toks.toFixed(1),
              p.itlAvg.toFixed(1),
              p.ttft.toFixed(2),
            ])}
          striped
        />
      </Stack>

      <Text size="small" tone="tertiary">
        Source: <Code>vllm/out/agentx_baseline_qwen38/&lt;arm&gt;/sa/seed42_c*</Code>{" "}
        · aiperf output_token_throughput.avg / inter_token_latency.avg
      </Text>
    </Stack>
  );
}
