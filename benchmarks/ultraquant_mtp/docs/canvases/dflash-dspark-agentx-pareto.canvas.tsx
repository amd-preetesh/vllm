import {
  BarChart,
  Callout,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Swatch,
  Table,
  Text,
  useHostTheme,
  useState,
} from "cursor/canvas";

type Kind = "base_fp8" | "fp8_fair" | "fp8_fair_fa" | "base_uq" | "best";

type Point = {
  conc: number;
  mode: string;
  kind: Kind;
  toks: number;
  itlAvg: number;
  ttft: number;
};

// AgentX seed42 · ~300s · Qwen3.8-27B TP=1 · KV8 / UQ / best · refreshed 2026-10-26 21:01 PDT.
const POINTS: Point[] = [
  {
    "conc": 8,
    "mode": "DSpark FP8",
    "kind": "base_fp8",
    "toks": 33.0,
    "itlAvg": 31.4,
    "ttft": 12.21
  },
  {
    "conc": 16,
    "mode": "DSpark FP8",
    "kind": "base_fp8",
    "toks": 27.7,
    "itlAvg": 48.1,
    "ttft": 6.34
  },
  {
    "conc": 20,
    "mode": "DSpark FP8",
    "kind": "base_fp8",
    "toks": 36.5,
    "itlAvg": 116.4,
    "ttft": 17.15
  },
  {
    "conc": 24,
    "mode": "DSpark FP8",
    "kind": "base_fp8",
    "toks": 24.8,
    "itlAvg": 135.9,
    "ttft": 16.29
  },
  {
    "conc": 30,
    "mode": "DSpark FP8",
    "kind": "base_fp8",
    "toks": 18.2,
    "itlAvg": 203.4,
    "ttft": 21.09
  },
  {
    "conc": 32,
    "mode": "DSpark FP8",
    "kind": "base_fp8",
    "toks": 6.8,
    "itlAvg": 428.6,
    "ttft": 68.78
  },
  {
    "conc": 8,
    "mode": "DSpark FP8 fair",
    "kind": "fp8_fair",
    "toks": 21.5,
    "itlAvg": 20.5,
    "ttft": 5.05
  },
  {
    "conc": 16,
    "mode": "DSpark FP8 fair",
    "kind": "fp8_fair",
    "toks": 44.5,
    "itlAvg": 24.1,
    "ttft": 3.7
  },
  {
    "conc": 20,
    "mode": "DSpark FP8 fair",
    "kind": "fp8_fair",
    "toks": 71.4,
    "itlAvg": 25.7,
    "ttft": 3.76
  },
  {
    "conc": 24,
    "mode": "DSpark FP8 fair",
    "kind": "fp8_fair",
    "toks": 60.9,
    "itlAvg": 34.9,
    "ttft": 3.15
  },
  {
    "conc": 30,
    "mode": "DSpark FP8 fair",
    "kind": "fp8_fair",
    "toks": 89.2,
    "itlAvg": 45.7,
    "ttft": 3.01
  },
  {
    "conc": 32,
    "mode": "DSpark FP8 fair",
    "kind": "fp8_fair",
    "toks": 74.2,
    "itlAvg": 59.6,
    "ttft": 3.71
  },
  {
    "conc": 8,
    "mode": "DSpark FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 21.5,
    "itlAvg": 22.2,
    "ttft": 3.39
  },
  {
    "conc": 16,
    "mode": "DSpark FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 44.5,
    "itlAvg": 22.4,
    "ttft": 2.48
  },
  {
    "conc": 20,
    "mode": "DSpark FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 76.2,
    "itlAvg": 24.9,
    "ttft": 2.14
  },
  {
    "conc": 24,
    "mode": "DSpark FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 65.8,
    "itlAvg": 25.5,
    "ttft": 2.92
  },
  {
    "conc": 30,
    "mode": "DSpark FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 94.0,
    "itlAvg": 30.9,
    "ttft": 2.44
  },
  {
    "conc": 32,
    "mode": "DSpark FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 105.2,
    "itlAvg": 37.3,
    "ttft": 1.99
  },
  {
    "conc": 8,
    "mode": "DSpark UQ",
    "kind": "base_uq",
    "toks": 21.5,
    "itlAvg": 22.1,
    "ttft": 9.91
  },
  {
    "conc": 16,
    "mode": "DSpark UQ",
    "kind": "base_uq",
    "toks": 44.5,
    "itlAvg": 24.8,
    "ttft": 5.19
  },
  {
    "conc": 20,
    "mode": "DSpark UQ",
    "kind": "base_uq",
    "toks": 49.8,
    "itlAvg": 46.3,
    "ttft": 8.63
  },
  {
    "conc": 24,
    "mode": "DSpark UQ",
    "kind": "base_uq",
    "toks": 52.5,
    "itlAvg": 61.5,
    "ttft": 7.63
  },
  {
    "conc": 30,
    "mode": "DSpark UQ",
    "kind": "base_uq",
    "toks": 56.7,
    "itlAvg": 107.4,
    "ttft": 17.58
  },
  {
    "conc": 32,
    "mode": "DSpark UQ",
    "kind": "base_uq",
    "toks": 17.4,
    "itlAvg": 231.2,
    "ttft": 43.51
  },
  {
    "conc": 8,
    "mode": "DSpark UQ opt",
    "kind": "best",
    "toks": 67.3,
    "itlAvg": 13.1,
    "ttft": 3.42
  },
  {
    "conc": 16,
    "mode": "DSpark UQ opt",
    "kind": "best",
    "toks": 55.9,
    "itlAvg": 15.0,
    "ttft": 2.47
  },
  {
    "conc": 20,
    "mode": "DSpark UQ opt",
    "kind": "best",
    "toks": 79.3,
    "itlAvg": 25.2,
    "ttft": 2.3
  },
  {
    "conc": 24,
    "mode": "DSpark UQ opt",
    "kind": "best",
    "toks": 68.6,
    "itlAvg": 23.5,
    "ttft": 3.16
  },
  {
    "conc": 30,
    "mode": "DSpark UQ opt",
    "kind": "best",
    "toks": 112.5,
    "itlAvg": 29.0,
    "ttft": 2.86
  },
  {
    "conc": 32,
    "mode": "DSpark UQ opt",
    "kind": "best",
    "toks": 105.2,
    "itlAvg": 36.4,
    "ttft": 2.37
  },
  {
    "conc": 8,
    "mode": "DFlash FP8",
    "kind": "base_fp8",
    "toks": 12.9,
    "itlAvg": 46.9,
    "ttft": 17.52
  },
  {
    "conc": 16,
    "mode": "DFlash FP8",
    "kind": "base_fp8",
    "toks": 26.6,
    "itlAvg": 77.4,
    "ttft": 6.26
  },
  {
    "conc": 20,
    "mode": "DFlash FP8",
    "kind": "base_fp8",
    "toks": 44.6,
    "itlAvg": 65.4,
    "ttft": 14.02
  },
  {
    "conc": 24,
    "mode": "DFlash FP8",
    "kind": "base_fp8",
    "toks": 43.6,
    "itlAvg": 78.6,
    "ttft": 15.28
  },
  {
    "conc": 30,
    "mode": "DFlash FP8",
    "kind": "base_fp8",
    "toks": 12.0,
    "itlAvg": 243.7,
    "ttft": 11.93
  },
  {
    "conc": 32,
    "mode": "DFlash FP8",
    "kind": "base_fp8",
    "toks": 11.8,
    "itlAvg": 339.1,
    "ttft": 21.41
  },
  {
    "conc": 8,
    "mode": "DFlash FP8 fair",
    "kind": "fp8_fair",
    "toks": 67.3,
    "itlAvg": 8.9,
    "ttft": 4.39
  },
  {
    "conc": 16,
    "mode": "DFlash FP8 fair",
    "kind": "fp8_fair",
    "toks": 101.7,
    "itlAvg": 12.0,
    "ttft": 3.21
  },
  {
    "conc": 20,
    "mode": "DFlash FP8 fair",
    "kind": "fp8_fair",
    "toks": 92.2,
    "itlAvg": 13.2,
    "ttft": 2.98
  },
  {
    "conc": 24,
    "mode": "DFlash FP8 fair",
    "kind": "fp8_fair",
    "toks": 175.2,
    "itlAvg": 15.6,
    "ttft": 3.51
  },
  {
    "conc": 30,
    "mode": "DFlash FP8 fair",
    "kind": "fp8_fair",
    "toks": 158.3,
    "itlAvg": 19.3,
    "ttft": 3.41
  },
  {
    "conc": 32,
    "mode": "DFlash FP8 fair",
    "kind": "fp8_fair",
    "toks": 174.6,
    "itlAvg": 31.5,
    "ttft": 3.42
  },
  {
    "conc": 8,
    "mode": "DFlash FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 67.3,
    "itlAvg": 8.9,
    "ttft": 2.88
  },
  {
    "conc": 16,
    "mode": "DFlash FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 132.3,
    "itlAvg": 11.0,
    "ttft": 2.31
  },
  {
    "conc": 20,
    "mode": "DFlash FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 155.7,
    "itlAvg": 10.7,
    "ttft": 1.91
  },
  {
    "conc": 24,
    "mode": "DFlash FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 149.4,
    "itlAvg": 11.9,
    "ttft": 2.48
  },
  {
    "conc": 30,
    "mode": "DFlash FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 163.4,
    "itlAvg": 17.3,
    "ttft": 2.3
  },
  {
    "conc": 32,
    "mode": "DFlash FP8 fair FA",
    "kind": "fp8_fair_fa",
    "toks": 185.3,
    "itlAvg": 16.9,
    "ttft": 2.12
  },
  {
    "conc": 8,
    "mode": "DFlash UQ",
    "kind": "base_uq",
    "toks": 13.2,
    "itlAvg": 49.9,
    "ttft": 11.65
  },
  {
    "conc": 16,
    "mode": "DFlash UQ",
    "kind": "base_uq",
    "toks": 27.7,
    "itlAvg": 57.4,
    "ttft": 6.74
  },
  {
    "conc": 20,
    "mode": "DFlash UQ",
    "kind": "base_uq",
    "toks": 27.3,
    "itlAvg": 115.6,
    "ttft": 7.43
  },
  {
    "conc": 24,
    "mode": "DFlash UQ",
    "kind": "base_uq",
    "toks": 27.0,
    "itlAvg": 131.1,
    "ttft": 7.43
  },
  {
    "conc": 30,
    "mode": "DFlash UQ",
    "kind": "base_uq",
    "toks": 16.3,
    "itlAvg": 210.8,
    "ttft": 10.52
  },
  {
    "conc": 32,
    "mode": "DFlash UQ",
    "kind": "base_uq",
    "toks": 12.3,
    "itlAvg": 297.9,
    "ttft": 22.52
  },
  {
    "conc": 8,
    "mode": "DFlash UQ SWA",
    "kind": "best",
    "toks": 67.3,
    "itlAvg": 13.4,
    "ttft": 7.41
  },
  {
    "conc": 16,
    "mode": "DFlash UQ SWA",
    "kind": "best",
    "toks": 101.7,
    "itlAvg": 18.2,
    "ttft": 4.19
  },
  {
    "conc": 20,
    "mode": "DFlash UQ SWA",
    "kind": "best",
    "toks": 129.4,
    "itlAvg": 22.0,
    "ttft": 4.44
  },
  {
    "conc": 24,
    "mode": "DFlash UQ SWA",
    "kind": "best",
    "toks": 101.6,
    "itlAvg": 22.1,
    "ttft": 5.63
  },
  {
    "conc": 30,
    "mode": "DFlash UQ SWA",
    "kind": "best",
    "toks": 103.3,
    "itlAvg": 37.9,
    "ttft": 5.88
  },
  {
    "conc": 32,
    "mode": "DFlash UQ SWA",
    "kind": "best",
    "toks": 115.0,
    "itlAvg": 45.4,
    "ttft": 6.61
  }
];

const CELL_ROWS = [
  ["DFlash FP8", "8", "12.9", "46.9", "17.52", "190884", "729"],
  ["DFlash FP8", "16", "26.6", "77.4", "6.26", "128668", "376"],
  ["DFlash FP8", "20", "44.6", "65.4", "14.02", "130446", "523"],
  ["DFlash FP8", "24", "43.6", "78.6", "15.28", "131864", "549"],
  ["DFlash FP8", "30", "12.0", "243.7", "11.93", "128481", "204"],
  ["DFlash FP8", "32", "11.8", "339.1", "21.41", "142777", "223"],
  ["DFlash FP8 fair", "8", "67.3", "8.9", "4.39", "188269", "2542"],
  ["DFlash FP8 fair", "16", "101.7", "12.0", "3.21", "118509", "864"],
  ["DFlash FP8 fair", "20", "92.2", "13.2", "2.98", "107296", "614"],
  ["DFlash FP8 fair", "24", "175.2", "15.6", "3.51", "123321", "1257"],
  ["DFlash FP8 fair", "30", "158.3", "19.3", "3.41", "128814", "718"],
  ["DFlash FP8 fair", "32", "174.6", "31.5", "3.42", "135705", "761"],
  ["DFlash FP8 fair FA", "8", "67.3", "8.9", "2.88", "188269", "2542"],
  ["DFlash FP8 fair FA", "16", "132.3", "11.0", "2.31", "119134", "1097"],
  ["DFlash FP8 fair FA", "20", "155.7", "10.7", "1.91", "108481", "963"],
  ["DFlash FP8 fair FA", "24", "149.4", "11.9", "2.48", "123793", "1129"],
  ["DFlash FP8 fair FA", "30", "163.4", "17.3", "2.3", "117441", "646"],
  ["DFlash FP8 fair FA", "32", "185.3", "16.9", "2.12", "133171", "768"],
  ["DFlash UQ", "8", "13.2", "49.9", "11.65", "190882", "729"],
  ["DFlash UQ", "16", "27.7", "57.4", "6.74", "126784", "349"],
  ["DFlash UQ", "20", "27.3", "115.6", "7.43", "134990", "423"],
  ["DFlash UQ", "24", "27.0", "131.1", "7.43", "136238", "437"],
  ["DFlash UQ", "30", "16.3", "210.8", "10.52", "129099", "241"],
  ["DFlash UQ", "32", "12.3", "297.9", "22.52", "143476", "209"],
  ["DFlash UQ SWA", "8", "67.3", "13.4", "7.41", "188268", "2542"],
  ["DFlash UQ SWA", "16", "101.7", "18.2", "4.19", "118509", "864"],
  ["DFlash UQ SWA", "20", "129.4", "22.0", "4.44", "109775", "871"],
  ["DFlash UQ SWA", "24", "101.6", "22.1", "5.63", "122103", "804"],
  ["DFlash UQ SWA", "30", "103.3", "37.9", "5.88", "132174", "540"],
  ["DFlash UQ SWA", "32", "115.0", "45.4", "6.61", "141747", "575"],
  ["DSpark FP8", "8", "33.0", "31.4", "12.21", "190883", "729"],
  ["DSpark FP8", "16", "27.7", "48.1", "6.34", "126783", "349"],
  ["DSpark FP8", "20", "36.5", "116.4", "17.15", "134035", "573"],
  ["DSpark FP8", "24", "24.8", "135.9", "16.29", "132510", "422"],
  ["DSpark FP8", "30", "18.2", "203.4", "21.09", "126500", "282"],
  ["DSpark FP8", "32", "6.8", "428.6", "68.78", "131901", "192"],
  ["DSpark FP8 fair", "8", "21.5", "20.5", "5.05", "185136", "914"],
  ["DSpark FP8 fair", "16", "44.5", "24.1", "3.7", "114031", "409"],
  ["DSpark FP8 fair", "20", "71.4", "25.7", "3.76", "110425", "592"],
  ["DSpark FP8 fair", "24", "60.9", "34.9", "3.15", "124380", "545"],
  ["DSpark FP8 fair", "30", "89.2", "45.7", "3.01", "136610", "514"],
  ["DSpark FP8 fair", "32", "74.2", "59.6", "3.71", "143019", "427"],
  ["DSpark FP8 fair FA", "8", "21.5", "22.2", "3.39", "185136", "914"],
  ["DSpark FP8 fair FA", "16", "44.5", "22.4", "2.48", "115476", "398"],
  ["DSpark FP8 fair FA", "20", "76.2", "24.9", "2.14", "108919", "576"],
  ["DSpark FP8 fair FA", "24", "65.8", "25.5", "2.92", "122556", "574"],
  ["DSpark FP8 fair FA", "30", "94.0", "30.9", "2.44", "134096", "484"],
  ["DSpark FP8 fair FA", "32", "105.2", "37.3", "1.99", "141702", "526"],
  ["DSpark UQ", "8", "21.5", "22.1", "9.91", "185135", "914"],
  ["DSpark UQ", "16", "44.5", "24.8", "5.19", "115477", "398"],
  ["DSpark UQ", "20", "49.8", "46.3", "8.63", "128712", "529"],
  ["DSpark UQ", "24", "52.5", "61.5", "7.63", "127871", "541"],
  ["DSpark UQ", "30", "56.7", "107.4", "17.58", "136479", "536"],
  ["DSpark UQ", "32", "17.4", "231.2", "43.51", "137872", "312"],
  ["DSpark UQ opt", "8", "67.3", "13.1", "3.42", "188270", "2542"],
  ["DSpark UQ opt", "16", "55.9", "15.0", "2.47", "116079", "487"],
  ["DSpark UQ opt", "20", "79.3", "25.2", "2.3", "107263", "550"],
  ["DSpark UQ opt", "24", "68.6", "23.5", "3.16", "123163", "569"],
  ["DSpark UQ opt", "30", "112.5", "29.0", "2.86", "131760", "562"],
  ["DSpark UQ opt", "32", "105.2", "36.4", "2.37", "141703", "526"]
];

const CELL_TONES = ["neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "success", "success", "success", "success", "success", "success", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "success", "success", "success", "success", "success", "success"] as const;

const DSPARK_MATRIX = [
  { name: "DSpark FP8", data: [33.0, 27.7, 36.5, 24.8, 18.2, 6.8] },
  { name: "DSpark FP8 fair", data: [21.5, 44.5, 71.4, 60.9, 89.2, 74.2] },
  { name: "DSpark FP8 fair FA", data: [21.5, 44.5, 76.2, 65.8, 94.0, 105.2] },
  { name: "DSpark UQ", data: [21.5, 44.5, 49.8, 52.5, 56.7, 17.4] },
  { name: "DSpark UQ opt", data: [67.3, 55.9, 79.3, 68.6, 112.5, 105.2] }
];

const DFLASH_MATRIX = [
  { name: "DFlash FP8", data: [12.9, 26.6, 44.6, 43.6, 12.0, 11.8] },
  { name: "DFlash FP8 fair", data: [67.3, 101.7, 92.2, 175.2, 158.3, 174.6] },
  { name: "DFlash FP8 fair FA", data: [67.3, 132.3, 155.7, 149.4, 163.4, 185.3] },
  { name: "DFlash UQ", data: [13.2, 27.7, 27.3, 27.0, 16.3, 12.3] },
  { name: "DFlash UQ SWA", data: [67.3, 101.7, 129.4, 101.6, 103.3, 115.0] }
];

function dominates(a: Point, b: Point): boolean {
  const betterOrEq = a.itlAvg <= b.itlAvg && a.toks >= b.toks;
  const strict = a.itlAvg < b.itlAvg || a.toks > b.toks;
  return betterOrEq && strict;
}

function frontier(points: Point[] = POINTS): Point[] {
  return points.filter(
    (p) => !points.some((q) => q !== p && dominates(q, p)),
  );
}

function pointsForFamily(family: "DFlash" | "DSpark"): Point[] {
  return POINTS.filter((p) => p.mode.startsWith(family));
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

const KIND_LABEL: Record<Kind, string> = {
  base_fp8: "FP8 KV8 · tgt UNIFIED · draft FA",
  fp8_fair: "FP8 fair · tgt UNIFIED · draft FA / FA_SWA",
  fp8_fair_fa: "FP8 fair FA · tgt FA · draft TRITON",
  base_uq: "UQ · tgt SPEC_COMPAT · draft FA_UQ_COMPAT",
  best: "Best · tgt SPEC_COMPAT · draft DENSE / FA_SWA",
};

const KIND_SWATCH: Record<Kind, "blue" | "cyan" | "purple" | "orange" | "green"> = {
  base_fp8: "blue",
  fp8_fair: "cyan",
  fp8_fair_fa: "purple",
  base_uq: "orange",
  best: "green",
};

const BASELINE_KINDS: Kind[] = ["base_fp8", "base_uq"];
const FAIR_BEST_KINDS: Kind[] = ["fp8_fair", "fp8_fair_fa", "best"];

function ParetoLegend({ kinds }: { kinds: Kind[] }) {
  return (
    <Row gap={16} wrap align="center">
      {kinds.map((k) => (
        <Row key={k} gap={6} align="center">
          <Swatch color={KIND_SWATCH[k]} style={{ width: 12, height: 12 }} />
          <Text size="small" tone="secondary">
            {KIND_LABEL[k]}
          </Text>
        </Row>
      ))}
      <Text size="small" tone="tertiary">
        dashed = frontier · dots labeled with C · hover for metrics
      </Text>
    </Row>
  );
}

function filterKinds(points: Point[], kinds: Kind[]): Point[] {
  return points.filter((p) => kinds.includes(p.kind));
}

function ParetoPlot({
  points,
  label,
}: {
  points: Point[];
  label: string;
}) {
  const theme = useHostTheme();
  const [hover, setHover] = useState<Point | null>(null);
  if (points.length === 0) {
    return (
      <Text tone="secondary">No points yet for {label}.</Text>
    );
  }
  const front = frontier(points);
  const xs = points.map((p) => p.itlAvg);
  const ys = points.map((p) => p.toks);
  const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.12 || 1;
  const yPad = (Math.max(...ys) - Math.min(...ys)) * 0.18 || 1;
  const xMin = Math.min(...xs) - xPad;
  const xMax = Math.max(...xs) + xPad;
  const yMin = Math.max(0, Math.min(...ys) - yPad);
  const yMax = Math.max(...ys) + yPad;

  const W = 860;
  const H = 480;
  const pad = { l: 72, r: 28, t: 18, b: 52 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const mapX = (x: number) => pad.l + ((x - xMin) / (xMax - xMin)) * iw;
  const mapY = (y: number) => pad.t + ((yMax - y) / (yMax - yMin)) * ih;

  const xTicks = ticks(xMin, xMax, 6);
  const yTicks = ticks(yMin, yMax, 5);
  const colors: Record<Kind, string> = {
    base_fp8: theme.category.blue,
    fp8_fair: theme.category.cyan,
    fp8_fair_fa: theme.category.purple,
    base_uq: theme.category.orange,
    best: theme.category.green,
  };

  const sortedFront = [...front].sort((a, b) => a.itlAvg - b.itlAvg);

  const byMode = new Map<string, Point[]>();
  for (const p of points) {
    const list = byMode.get(p.mode) ?? [];
    list.push(p);
    byMode.set(p.mode, list);
  }
  const seriesPaths = [...byMode.entries()]
    .map(([mode, pts]) => {
      const sorted = [...pts].sort((a, b) => a.conc - b.conc);
      return { mode, kind: sorted[0].kind, pts: sorted };
    })
    .filter((s) => s.pts.length >= 2);

  const tipW = 220;
  const tipH = 58;
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
      aria-label={`${label} AgentX Pareto: mean ITL vs output throughput`}
      onMouseLeave={() => setHover(null)}
    >
      {xTicks.map((t) => (
        <g key={`x${t}`}>
          <line
            x1={mapX(t)}
            x2={mapX(t)}
            y1={pad.t}
            y2={pad.t + ih}
            stroke={theme.stroke.tertiary}
          />
          <text
            x={mapX(t)}
            y={pad.t + ih + 18}
            textAnchor="middle"
            fill={theme.text.secondary}
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
            stroke={theme.stroke.tertiary}
          />
          <text
            x={pad.l - 10}
            y={mapY(t) + 4}
            textAnchor="end"
            fill={theme.text.secondary}
            fontSize={11}
          >
            {fmt(t)}
          </text>
        </g>
      ))}
      {seriesPaths.map((s) => (
        <polyline
          key={`series-${s.mode}`}
          fill="none"
          stroke={colors[s.kind]}
          strokeWidth={1.5}
          strokeOpacity={0.45}
          points={s.pts
            .map((p) => `${mapX(p.itlAvg)},${mapY(p.toks)}`)
            .join(" ")}
        />
      ))}
      {sortedFront.length > 1 ? (
        <polyline
          fill="none"
          stroke={theme.accent.primary}
          strokeWidth={2.5}
          strokeDasharray="6 4"
          points={sortedFront
            .map((f) => `${mapX(f.itlAvg)},${mapY(f.toks)}`)
            .join(" ")}
        />
      ) : null}
      {points.map((p) => {
        const on = front.includes(p);
        const active = hover === p;
        return (
          <g
            key={`${p.mode}-c${p.conc}`}
            onMouseEnter={() => setHover(p)}
            style={{ cursor: "pointer" }}
          >
            <circle
              cx={mapX(p.itlAvg)}
              cy={mapY(p.toks)}
              r={14}
              fill="transparent"
            />
            <circle
              cx={mapX(p.itlAvg)}
              cy={mapY(p.toks)}
              r={active ? 8 : on ? 7 : 5}
              fill={colors[p.kind]}
              stroke={
                active || on ? theme.accent.primary : theme.stroke.primary
              }
              strokeWidth={active || on ? 2 : 1}
            />
            <text
              x={mapX(p.itlAvg) + 9}
              y={mapY(p.toks) - 8}
              fill={theme.text.secondary}
              fontSize={10}
              fontWeight={600}
            >
              {String(p.conc)}
            </text>
          </g>
        );
      })}
      {hover ? (
        <g pointerEvents="none">
          <rect
            x={tipX}
            y={tipY}
            width={tipW}
            height={tipH}
            rx={6}
            fill={theme.bg.elevated}
            stroke={theme.stroke.secondary}
          />
          <text
            x={tipX + 10}
            y={tipY + 18}
            fill={theme.text.primary}
            fontSize={12}
            fontWeight={600}
          >
            {`c${hover.conc} · ${hover.mode}`}
          </text>
          <text
            x={tipX + 10}
            y={tipY + 36}
            fill={theme.text.secondary}
            fontSize={11}
          >
            {`${hover.toks} tok/s · ITL ${hover.itlAvg} ms`}
          </text>
          <text
            x={tipX + 10}
            y={tipY + 50}
            fill={theme.text.tertiary}
            fontSize={11}
          >
            {`TTFT ${hover.ttft}s · ${KIND_LABEL[hover.kind]}`}
          </text>
        </g>
      ) : null}
      <text
        x={pad.l + iw / 2}
        y={H - 8}
        textAnchor="middle"
        fill={theme.text.secondary}
        fontSize={12}
      >
        Mean ITL (ms) — lower is better
      </text>
      <text
        x={16}
        y={pad.t + ih / 2}
        textAnchor="middle"
        fill={theme.text.secondary}
        fontSize={12}
        transform={`rotate(-90 16 ${pad.t + ih / 2})`}
      >
        Output tok/s — higher is better
      </text>
    </svg>
  );
}

function frontierRows(points: Point[]) {
  return frontier(points)
    .slice()
    .sort((a, b) => b.toks - a.toks)
    .map((p) => [
      p.mode,
      String(p.conc),
      String(p.toks),
      String(p.itlAvg),
      String(p.ttft),
    ]);
}

export default function DflashDsparkAgentxPareto() {
  const dspark = pointsForFamily("DSpark");
  const dflash = pointsForFamily("DFlash");
  return (
    <Stack gap={28}>
      <Stack gap={6}>
        <H1>DFlash / DSpark AgentX Pareto</H1>
        <Text tone="secondary">
          60 cells · DSpark / DFlash · baselines on their own axes · FP8 fair
          + best UQ on a second plot · Qwen3.8-27B · TP=1 · AgentX seed42 ·
          ~300s · refreshed 2026-10-26 21:01 PDT
        </Text>
        <Row gap={8}>
          <Pill tone="info">FP8 KV8</Pill>
          <Pill>UQ baseline</Pill>
          <Pill tone="info">FP8 fair</Pill>
          <Pill tone="success">Best: DSpark opt · DFlash SWA</Pill>
        </Row>
      </Stack>

      <Grid columns={3} gap={16}>
        <Stat
          value="112.5"
          label="DSpark UQ opt c30"
          tone="success"
        />
        <Stat
          value="129.4"
          label="DFlash UQ SWA c20"
          tone="success"
        />
        <Stat
          value="36.5 / 44.6"
          label="FP8 KV8 peaks (DS / DF)"
        />
      </Grid>

      <Callout tone="info" title="Plot layout + attention backends">
        <Code>Baselines</Code> = FP8 KV8 (tgt UNIFIED, draft FA) + UQ (tgt
        SPEC_COMPAT, draft FA_UQ_COMPAT). <Code>Fair + best</Code> = FP8 fair
        UNIFIED (draft FA / FA_SWA) + FP8 fair FA when present (tgt FA, draft
        TRITON) + best UQ (tgt SPEC_COMPAT; draft FA_UQ_DENSE or FA_DFLASH_SWA).
        Own axes so the baseline eviction tail does not squash fair/best. UQ
        peaks: DSpark 56.7 · DFlash 27.7.
      </Callout>

      <Stack gap={10}>
        <H2>Pareto: output tok/s vs mean ITL</H2>
        <Text tone="secondary">
          Solid lines = concurrency path; dashed = frontier on that plot. Dot
          labels = C (8…32). Hover for config / metrics.
        </Text>
      </Stack>

      <Stack gap={10}>
        <H3>DSpark — baselines (FP8 KV8 + UQ)</H3>
        <ParetoLegend kinds={BASELINE_KINDS} />
        <ParetoPlot
          points={filterKinds(dspark, BASELINE_KINDS)}
          label="DSpark baselines"
        />
        <Table
          headers={["Config", "Conc", "tok/s", "ITL ms", "TTFT s"]}
          columnAlign={["left", "right", "right", "right", "right"]}
          rows={frontierRows(filterKinds(dspark, BASELINE_KINDS))}
          striped
        />
      </Stack>

      <Stack gap={10}>
        <H3>DSpark — FP8 fair + best UQ</H3>
        <ParetoLegend kinds={FAIR_BEST_KINDS} />
        <ParetoPlot
          points={filterKinds(dspark, FAIR_BEST_KINDS)}
          label="DSpark fair + best"
        />
        <Table
          headers={["Config", "Conc", "tok/s", "ITL ms", "TTFT s"]}
          columnAlign={["left", "right", "right", "right", "right"]}
          rows={frontierRows(filterKinds(dspark, FAIR_BEST_KINDS))}
          striped
        />
      </Stack>

      <Stack gap={10}>
        <H3>DFlash — baselines (FP8 KV8 + UQ)</H3>
        <ParetoLegend kinds={BASELINE_KINDS} />
        <ParetoPlot
          points={filterKinds(dflash, BASELINE_KINDS)}
          label="DFlash baselines"
        />
        <Table
          headers={["Config", "Conc", "tok/s", "ITL ms", "TTFT s"]}
          columnAlign={["left", "right", "right", "right", "right"]}
          rows={frontierRows(filterKinds(dflash, BASELINE_KINDS))}
          striped
        />
      </Stack>

      <Stack gap={10}>
        <H3>DFlash — FP8 fair + best UQ</H3>
        <ParetoLegend kinds={FAIR_BEST_KINDS} />
        <ParetoPlot
          points={filterKinds(dflash, FAIR_BEST_KINDS)}
          label="DFlash fair + best"
        />
        <Table
          headers={["Config", "Conc", "tok/s", "ITL ms", "TTFT s"]}
          columnAlign={["left", "right", "right", "right", "right"]}
          rows={frontierRows(filterKinds(dflash, FAIR_BEST_KINDS))}
          striped
        />
        <Text size="small" tone="tertiary">
          Source: <Code>vllm/out/*_qwen38_*/sa/seed42_c*/</Code> · aiperf
          output_token_throughput / inter_token_latency (ms).
        </Text>
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>KV eviction vs headroom (DFlash)</H2>
        <Callout tone="warning" title="DFlash FP8 KV8 eviction knee at C=30 (not attention kernel)">
          Throughput collapses between c24 (43.6 tok/s, ITL 78.6 ms, hit 43.7%, KV max 29.0%) and c30 (12.0 tok/s, ITL 243.7 ms, hit 36.9%, KV max 44.0%). Prefix hit rate falls as concurrency rises while preemptions stay 0 — classic prefix-cache eviction / pool pressure. ROCM_AITER_FA on target would not fix this; fair runs already use UNIFIED on target + draft FA/SWA + ret≈50k.
        </Callout>
        <Table
          headers={[
            "Config",
            "C",
            "tok/s",
            "ITL ms",
            "hit %",
            "KV avg %",
            "KV max %",
          ]}
          columnAlign={[
            "left",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
          ]}
          rows={[
  ["DFlash FP8 KV8", "8", "12.9", "46.9", "55.3", "7.7", "13.6"],
  ["DFlash FP8 KV8", "16", "26.6", "77.4", "57.7", "11.5", "20.0"],
  ["DFlash FP8 KV8", "20", "44.6", "65.4", "49.8", "15.4", "28.9"],
  ["DFlash FP8 KV8", "24", "43.6", "78.6", "43.7", "17.8", "29.0"],
  ["DFlash FP8 KV8", "30", "12.0", "243.7", "36.9", "28.6", "44.0"],
  ["DFlash FP8 KV8", "32", "11.8", "339.1", "32.0", "31.3", "49.0"],
  ["DFlash FP8 fair", "8", "67.3", "8.9", "68.7", "5.1", "9.8"],
  ["DFlash FP8 fair", "16", "101.7", "12.0", "67.1", "7.8", "13.6"],
  ["DFlash FP8 fair", "20", "92.2", "13.2", "64.6", "9.1", "17.2"],
  ["DFlash FP8 fair", "24", "175.2", "15.6", "61.5", "10.0", "17.3"],
  ["DFlash FP8 fair", "30", "158.3", "19.3", "69.2", "15.4", "25.5"],
  ["DFlash FP8 fair", "32", "174.6", "31.5", "70.6", "18.3", "39.6"]
          ]}
          striped
        />
        <Callout tone="success" title="UQ+DFlash SWA at C=32: KV max 31.8% (~68% free), hit 63.2%">
          At C=32, UQ+DFlash SWA holds 115.0 tok/s / ITL 45.4 ms with KV avg 18.2% / max 31.8% and prefix hit 63.2% (preemptions 0). Stays healthy via SWA draft window + ret≈50k + denser UQ KV pool (less eviction than FP8 KV8). FP8 KV8 c32: 11.8 tok/s, hit 32.0%, KV max 49.0%; UQ baseline c32: 12.3 tok/s, hit 24.3%, KV max 26.3%
        </Callout>
        <Table
          headers={[
            "Config",
            "C",
            "tok/s",
            "ITL ms",
            "hit %",
            "KV avg %",
            "KV max %",
          ]}
          columnAlign={[
            "left",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
          ]}
          rows={[
  ["DFlash UQ", "8", "13.2", "49.9", "47.4", "4.1", "6.1"],
  ["DFlash UQ", "16", "27.7", "57.4", "55.7", "16.9", "28.5"],
  ["DFlash UQ", "20", "27.3", "115.6", "37.5", "9.3", "15.4"],
  ["DFlash UQ", "24", "27.0", "131.1", "33.7", "10.9", "17.2"],
  ["DFlash UQ", "30", "16.3", "210.8", "32.6", "15.8", "23.3"],
  ["DFlash UQ", "32", "12.3", "297.9", "24.3", "15.8", "26.3"],
  ["DFlash UQ SWA", "8", "67.3", "13.4", "58.0", "4.1", "8.3"],
  ["DFlash UQ SWA", "16", "101.7", "18.2", "59.2", "6.3", "13.5"],
  ["DFlash UQ SWA", "20", "129.4", "22.0", "55.1", "8.7", "19.6"],
  ["DFlash UQ SWA", "24", "101.6", "22.1", "52.0", "10.2", "19.8"],
  ["DFlash UQ SWA", "30", "103.3", "37.9", "60.7", "15.4", "27.4"],
  ["DFlash UQ SWA", "32", "115.0", "45.4", "63.2", "18.2", "31.8"]
          ]}
          striped
        />
        <Text size="small" tone="tertiary">
          KV % from aiperf <Code>vllm:kv_cache_usage_perc</Code> (avg/max over
          run). Hit % from <Code>prefix_cache_hits/queries</Code> in 
          <Code>metrics_end.txt</Code> (server.log fallback). Preemptions were 0
          across these cells.
        </Text>
      </Stack>

      <Divider />

      <Stack gap={10}>
        <H2>Throughput vs concurrency</H2>
        <H3>DSpark — tok/s by conc</H3>
        <BarChart
          height={280}
          categories={["c8", "c16", "c20", "c24", "c30", "c32"]}
          valueSuffix=" tok/s"
          series={DSPARK_MATRIX}
        />
        <H3>DFlash — tok/s by conc</H3>
        <BarChart
          height={280}
          categories={["c8", "c16", "c20", "c24", "c30", "c32"]}
          valueSuffix=" tok/s"
          series={DFLASH_MATRIX}
        />
        <Text size="small" tone="tertiary">
          Missing conc cells plot as 0 (DFlash SWA is c32-only so far).
        </Text>
      </Stack>

      <Stack gap={10}>
        <H2>All cells</H2>
        <Table
          headers={[
            "Config",
            "Conc",
            "tok/s",
            "ITL ms",
            "TTFT s",
            "ISL",
            "OSL",
          ]}
          columnAlign={[
            "left",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
          ]}
          rows={CELL_ROWS}
          rowTone={[...CELL_TONES]}
          striped
          stickyHeader
        />
      </Stack>
    </Stack>
  );
}
