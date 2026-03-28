#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


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
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt_num(x: float, decimals: int = 4) -> str:
    if x is None or (
        isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    ):
        return "—"
    return f"{x:.{decimals}f}"


def fmt_speedup(x: float, decimals: int = 2) -> str:
    if x is None or (
        isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    ):
        return "—"
    return f"{x:.{decimals}f}x"


@dataclass
class Config:
    csv: Optional[Path]
    out_md: Path
    out_dir: Path
    title: str


def canonical_variant_order(cols: List[str]) -> List[str]:
    order = ["baseline", "opt1", "opt2", "opt3", "opt4"]
    present = [c for c in order if c in cols]
    extras = [c for c in cols if c not in present]
    return present + extras


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in ("variant", "n", "time_ms"):
        if col not in df.columns:
            raise ValueError(f"CSV must have a '{col}' column.")

    df["variant"] = df["variant"].astype(str)
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")

    if "cpu_time_ms" in df.columns:
        df["cpu_time_ms"] = pd.to_numeric(df["cpu_time_ms"], errors="coerce")

    df = df.dropna(subset=["variant", "n", "time_ms"]).copy()
    df["n"] = df["n"].astype(int)
    return df


def compute_speedups(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = ["time_ms"]
    if "cpu_time_ms" in df.columns:
        agg_cols.append("cpu_time_ms")

    g = df.groupby(["variant", "n"], as_index=False)[agg_cols].mean()
    baseline = (
        g[g["variant"] == "baseline"][["n", "time_ms"]]
        .rename(columns={"time_ms": "baseline_ms"})
    )

    out = g.merge(baseline, on="n", how="left")
    out["speedup"] = out["baseline_ms"] / out["time_ms"]
    return out


def pivot_table_time(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.pivot_table(
        index="n",
        columns="variant",
        values="time_ms",
        aggfunc="mean",
    ).sort_index()
    piv = piv[canonical_variant_order(list(piv.columns))]
    piv.index.name = "n"

    if "cpu_time_ms" in df.columns:
        cpu_by_n = df.groupby("n")["cpu_time_ms"].mean()
        piv.insert(0, "cpu_ref", cpu_by_n)
    return piv


def pivot_table_speedup(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.pivot_table(
        index="n",
        columns="variant",
        values="speedup",
        aggfunc="mean",
    ).sort_index()
    piv = piv[canonical_variant_order(list(piv.columns))]
    piv.index.name = "n"
    return piv


def save_plot_time_vs_n(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return

    plt.figure()

    if "cpu_time_ms" in df.columns:
        cpu = (
            df.groupby("n")["cpu_time_ms"]
            .mean()
            .reset_index()
            .sort_values("n")
        )
        cpu = cpu.dropna(subset=["cpu_time_ms"])
        if not cpu.empty:
            plt.plot(
                cpu["n"],
                cpu["cpu_time_ms"],
                marker="s",
                linestyle="--",
                color="gray",
                label="cpu_ref",
            )

    for variant in canonical_variant_order(
        sorted(df["variant"].unique().tolist())
    ):
        d = df[df["variant"] == variant].sort_values("n")
        if d.empty:
            continue
        plt.plot(d["n"], d["time_ms"], marker="o", label=variant)

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("n (input elements)")
    plt.ylabel("avg time per iteration (ms)")
    plt.title("Sort: time vs input size")
    plt.grid(True)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def save_plot_speedup_vs_n(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return

    plt.figure()
    for variant in canonical_variant_order(
        sorted(df["variant"].unique().tolist())
    ):
        if variant == "baseline":
            continue
        d = df[df["variant"] == variant].sort_values("n")
        if d.empty:
            continue
        plt.plot(d["n"], d["speedup"], marker="o", label=variant)

    plt.xscale("log", base=2)
    plt.xlabel("n (input elements)")
    plt.ylabel("speedup vs baseline")
    plt.title("Sort: speedup vs baseline")
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
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

    time_plot = cfg.out_dir / "sort_time.png"
    speed_plot = cfg.out_dir / "sort_speedup.png"
    save_plot_time_vs_n(df_agg, time_plot)
    save_plot_speedup_vs_n(df_agg, speed_plot)

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
    md.append("### Time vs input size\n")
    md.append(f"![time vs n]({time_plot.name})\n")
    md.append("### Speedup vs baseline\n")
    md.append(f"![speedup vs baseline]({speed_plot.name})\n")

    md.append("## Tables\n")
    md.append("> Notes:\n")
    md.append(
        "> - Variants: baseline (1-bit radix), opt1 (coalesced radix), "
        "opt2 (coarsened radix), opt3 (bitonic sort).\n"
    )
    md.append(
        "> - `cpu_ref` is the single-threaded CPU reference (`std::sort`, "
        "not a GPU variant).\n"
    )
    md.append("> - Speedup is computed as `baseline_time / variant_time`.\n")
    md.append(
        "> - If a row shows `—`, it usually means baseline timing is "
        "missing for that size.\n"
    )

    t = pivot_table_time(df_agg)
    s = pivot_table_speedup(df_agg)

    headers = [t.index.name] + list(t.columns)
    rows = []
    for idx, row in t.iterrows():
        rows.append(
            [str(int(idx))]
            + [
                fmt_num(float(row[c])) if pd.notna(row[c]) else "—"
                for c in t.columns
            ]
        )
    md.append("\n**Avg time per iteration (ms)**\n")
    md.append(md_table(headers, rows) + "\n")

    headers = [s.index.name] + list(s.columns)
    rows = []
    for idx, row in s.iterrows():
        rows.append(
            [str(int(idx))]
            + [
                fmt_speedup(float(row[c])) if pd.notna(row[c]) else "—"
                for c in s.columns
            ]
        )
    md.append("\n**Speedup vs baseline**\n")
    md.append(md_table(headers, rows) + "\n")

    cfg.out_md.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    root = repo_root_from_this_file()
    results_dir = root / "benchmarks" / "results"

    ap = argparse.ArgumentParser(
        description=(
            "Generate sort benchmark markdown + plots from latest CSV."
        )
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Path to CSV. If omitted, auto-picks latest sort_*.csv "
            "in benchmarks/results/."
        ),
    )
    pat_dir = root / "docs" / "plots" / "sort"
    ap.add_argument(
        "--out-md",
        type=str,
        default=str(pat_dir / "sort_results.md"),
        help="Output markdown file path.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(pat_dir),
        help="Directory for output plots.",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="Sort Benchmark Results",
        help="Markdown title.",
    )
    args = ap.parse_args()

    cfg = Config(
        csv=Path(args.csv) if args.csv else None,
        out_md=Path(args.out_md),
        out_dir=Path(args.out_dir),
        title=args.title,
    )

    if not results_dir.exists():
        raise SystemExit(f"Missing results directory: {results_dir}")

    if cfg.csv:
        csv_path = cfg.csv
    else:
        csv_path = latest_file(results_dir, "sort_*.csv")
        if csv_path is None:
            csv_path = latest_file(results_dir, "bench_*.csv")

    if csv_path is None or not csv_path.exists():
        raise SystemExit(
            "No benchmark CSV found. Run: bash scripts/bench_sort.sh"
        )

    df = load_csv(csv_path)

    if "pattern" in df.columns:
        sort_like = df[
            df["pattern"].astype(str).str.contains(
                "sort",
                case=False,
                regex=True,
            )
        ]
        if not sort_like.empty:
            df = sort_like.copy()

    df_agg = compute_speedups(df)
    write_markdown(cfg, df_agg, csv_path)

    print(f"Wrote: {cfg.out_md}")
    print(f"Plots: {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
