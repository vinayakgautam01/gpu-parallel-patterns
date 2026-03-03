#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def latest_file(glob_path: Path, pattern: str) -> Optional[Path]:
    files = list(glob_path.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def try_git_rev(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def fmt_num(x, decimals=4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{decimals}f}"


def fmt_speedup(x, decimals=2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{decimals}f}×"


def is_perfect_square(n: int) -> Optional[int]:
    if n < 0:
        return None
    s = int(math.isqrt(n))
    return s if s * s == n else None


# -----------------------------
# Core
# -----------------------------

@dataclass
class Config:
    csv: Optional[Path]
    out_md: Path
    out_dir: Path
    r_values: List[int]
    r_auto_count: int
    n_for_speedup_plot: Optional[int]
    title: str


def pick_r_values(df: pd.DataFrame, requested: List[int], auto_count: int) -> List[int]:
    rs = sorted({int(r) for r in df["R"].dropna().unique().tolist()})
    if not rs:
        return [1]

    if requested:
        requested = [int(r) for r in requested]
        keep = [r for r in requested if r in rs]
        return keep if keep else [rs[0]]

    preferred = [1, 2, 3, 5, 8, 12, 15, 31]
    picked = []
    for r in preferred:
        if r in rs and r not in picked:
            picked.append(r)
        if len(picked) >= auto_count:
            break

    if len(picked) < auto_count:
        for r in rs:
            if r not in picked:
                picked.append(r)
            if len(picked) >= auto_count:
                break

    return picked[:auto_count]


def canonical_variant_order(cols: List[str]) -> List[str]:
    order = ["baseline", "opt1", "opt2", "opt3", "opt4"]
    present = [c for c in order if c in cols]
    extras = [c for c in cols if c not in present]
    return present + extras


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "variant" not in df.columns:
        raise ValueError("CSV must have a 'variant' column.")
    if "n" not in df.columns:
        raise ValueError("CSV must have an 'n' column.")
    if "time_ms" not in df.columns:
        raise ValueError("CSV must have a 'time_ms' column.")

    if "pattern" not in df.columns:
        df["pattern"] = "convolution"
    if "R" not in df.columns:
        df["R"] = 1

    df["variant"] = df["variant"].astype(str)
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["R"] = pd.to_numeric(df["R"], errors="coerce")
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")

    df = df.dropna(subset=["variant", "n", "R", "time_ms"]).copy()
    df["n"] = df["n"].astype(int)
    df["R"] = df["R"].astype(int)

    if "w" not in df.columns or "h" not in df.columns:
        ws = []
        hs = []
        for n in df["n"].tolist():
            s = is_perfect_square(int(n))
            if s is None:
                ws.append(None)
                hs.append(None)
            else:
                ws.append(s)
                hs.append(s)
        df["w"] = ws
        df["h"] = hs
    else:
        df["w"] = pd.to_numeric(df["w"], errors="coerce")
        df["h"] = pd.to_numeric(df["h"], errors="coerce")

    return df


def compute_speedups(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = ["variant", "n", "R"]
    if "w" in df.columns:
        agg_cols.append("w")
    if "h" in df.columns:
        agg_cols.append("h")

    g = df.groupby(agg_cols, as_index=False)["time_ms"].mean()

    baseline = (
        g[g["variant"] == "baseline"][["n", "R", "time_ms"]]
        .rename(columns={"time_ms": "baseline_ms"})
    )

    out = g.merge(baseline, on=["n", "R"], how="left")
    out["speedup"] = out["baseline_ms"] / out["time_ms"]
    return out


def pivot_table_time(df: pd.DataFrame, R: int) -> pd.DataFrame:
    sub = df[df["R"] == R].copy()
    if sub["w"].notna().all():
        idx = "w"
    else:
        idx = "n"

    piv = sub.pivot_table(index=idx, columns="variant", values="time_ms", aggfunc="mean").sort_index()
    piv = piv[canonical_variant_order(list(piv.columns))]
    piv.index.name = idx
    return piv


def pivot_table_speedup(df: pd.DataFrame, R: int) -> pd.DataFrame:
    sub = df[df["R"] == R].copy()
    if sub["w"].notna().all():
        idx = "w"
    else:
        idx = "n"

    piv = sub.pivot_table(index=idx, columns="variant", values="speedup", aggfunc="mean").sort_index()
    piv = piv[canonical_variant_order(list(piv.columns))]
    piv.index.name = idx
    return piv


def save_plot_time_vs_size(df: pd.DataFrame, R: int, out_png: Path) -> None:
    sub = df[df["R"] == R].copy()
    if sub.empty:
        return

    xcol = "w" if sub["w"].notna().all() else "n"

    plt.figure()
    for v in canonical_variant_order(sorted(sub["variant"].unique().tolist())):
        d = sub[sub["variant"] == v].sort_values(xcol)
        if d.empty:
            continue
        plt.plot(d[xcol], d["time_ms"], marker="o", label=v)

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("image side w (square)" if xcol == "w" else "n (= w*h)")
    plt.ylabel("avg time per iteration (ms)")
    plt.title(f"Convolution: time vs size (R={R})")
    plt.grid(True)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def save_plot_speedup_vs_R(df: pd.DataFrame, n0: int, out_png: Path) -> None:
    sub = df[df["n"] == n0].copy()
    if sub.empty:
        return

    plt.figure()
    for v in canonical_variant_order(sorted(sub["variant"].unique().tolist())):
        if v == "baseline":
            continue
        d = sub[sub["variant"] == v].sort_values("R")
        if d.empty:
            continue
        plt.plot(d["R"], d["speedup"], marker="o", label=v)

    plt.xlabel("R (radius)")
    plt.ylabel("speedup vs baseline")
    plt.title(f"Convolution: speedup vs R (n={n0})")
    plt.grid(True)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def write_markdown(cfg: Config, df_agg: pd.DataFrame, csv_path: Path) -> None:
    root = repo_root_from_this_file()
    git_rev = try_git_rev(root)

    cfg.out_md.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    r_vals = pick_r_values(df_agg, cfg.r_values, cfg.r_auto_count)

    n_max = int(df_agg["n"].max()) if cfg.n_for_speedup_plot is None else int(cfg.n_for_speedup_plot)

    time_plot = cfg.out_dir / f"conv_time_R{r_vals[0]}.png"
    speed_plot = cfg.out_dir / f"conv_speedup_n{n_max}.png"

    save_plot_time_vs_size(df_agg, r_vals[0], time_plot)
    save_plot_speedup_vs_R(df_agg, n_max, speed_plot)

    env_path = csv_path.with_suffix("").with_name(csv_path.stem + "_env.txt")
    env_note = ""
    if env_path.exists():
        env_note = f"- Environment capture: `{env_path.as_posix()}`\n"

    md = []
    md.append(f"# {cfg.title}\n")
    md.append(f"- Generated from: `{csv_path.as_posix()}`\n")
    md.append(f"- Git revision: `{git_rev}`\n")
    md.append(env_note)

    md.append("## Plots\n")
    md.append(f"### Time vs size (R={r_vals[0]})\n")
    md.append(f"![time vs size]({time_plot.name})\n")
    md.append(f"### Speedup vs R (n={n_max})\n")
    md.append(f"![speedup vs R]({speed_plot.name})\n")

    md.append("## Tables\n")
    md.append("> Notes:\n")
    md.append("> - Speedup is computed as `baseline_time / variant_time`.\n")
    md.append("> - If a row shows `—`, it usually means baseline timing is missing for that (n,R).\n")

    for R in r_vals:
        md.append(f"\n### R = {R}\n")

        t = pivot_table_time(df_agg, R)
        s = pivot_table_speedup(df_agg, R)

        headers = [t.index.name] + list(t.columns)
        rows = []
        for idx, row in t.iterrows():
            rows.append([str(int(idx))] + [fmt_num(float(row[c])) if pd.notna(row[c]) else "—" for c in t.columns])
        md.append("**Avg time per iteration (ms)**\n")
        md.append(md_table(headers, rows) + "\n")

        headers = [s.index.name] + list(s.columns)
        rows = []
        for idx, row in s.iterrows():
            rows.append([str(int(idx))] + [fmt_speedup(float(row[c])) if pd.notna(row[c]) else "—" for c in s.columns])
        md.append("**Speedup vs baseline**\n")
        md.append(md_table(headers, rows) + "\n")

    cfg.out_md.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    root = repo_root_from_this_file()
    results_dir = root / "benchmarks" / "results"

    ap = argparse.ArgumentParser(description="Generate convolution benchmark markdown + plots from latest CSV.")
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to CSV. If omitted, auto-picks latest conv_*.csv in benchmarks/results/.")
    pat_dir = root / "docs" / "plots" / "convolution"
    ap.add_argument("--out-md", type=str,
                    default=str(pat_dir / "convolution_results.md"),
                    help="Output markdown file path.")
    ap.add_argument("--out-dir", type=str,
                    default=str(pat_dir),
                    help="Directory for output plots.")
    ap.add_argument("--r", dest="r_values", action="append", default=[],
                    help="R value to include in tables (repeatable). If omitted, auto-picks a few.")
    ap.add_argument("--r-auto-count", type=int, default=3,
                    help="How many R values to auto-pick when --r not provided.")
    ap.add_argument("--n-for-speedup-plot", type=int, default=None,
                    help="n (w*h) to use for speedup-vs-R plot. Default: max n present.")
    ap.add_argument("--title", type=str, default="Convolution Benchmark Results",
                    help="Markdown title.")
    args = ap.parse_args()

    cfg = Config(
        csv=Path(args.csv) if args.csv else None,
        out_md=Path(args.out_md),
        out_dir=Path(args.out_dir),
        r_values=[int(x) for x in args.r_values] if args.r_values else [],
        r_auto_count=int(args.r_auto_count),
        n_for_speedup_plot=args.n_for_speedup_plot,
        title=args.title,
    )

    if not results_dir.exists():
        raise SystemExit(f"Missing results directory: {results_dir}")

    csv_path: Optional[Path]
    if cfg.csv:
        csv_path = cfg.csv
    else:
        csv_path = latest_file(results_dir, "conv_*.csv")
        if csv_path is None:
            csv_path = latest_file(results_dir, "bench_*.csv")

    if csv_path is None or not csv_path.exists():
        raise SystemExit("No benchmark CSV found. Run: bash scripts/bench_conv.sh")

    df = load_csv(csv_path)

    if "pattern" in df.columns:
        conv_like = df[df["pattern"].astype(str).str.contains("conv|convolution", case=False, regex=True)]
        if not conv_like.empty:
            df = conv_like.copy()

    df_agg = compute_speedups(df)

    write_markdown(cfg, df_agg, csv_path)

    print(f"Wrote: {cfg.out_md}")
    print(f"Plots: {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
