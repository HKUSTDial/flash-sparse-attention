from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import List, Optional

from test_utils import BenchmarkResult

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

_HATCHES = ["///", "xx", "\\\\", "..", "++", "OO", "**", "||"]


def _field_to_label(field_name: str) -> str:
    """``triton_dense_ms`` → ``Triton Dense``, ``fa_dense_ms`` → ``FA Dense``."""
    parts = field_name.removesuffix("_ms").split("_")
    return " ".join(
        p.upper() if p in ("fa", "cudnn") else p.capitalize() for p in parts
    )


def _discover_active_series(ok: list[BenchmarkResult]):
    """Discover series from dataclass fields, return [(label, [vals])]."""
    ms_fields = [
        f.name for f in dataclasses.fields(BenchmarkResult) if f.name.endswith("_ms")
    ]
    # non-triton (fa, cudnn) first, then triton; preserve declaration order within
    ms_fields.sort(key=lambda n: (n.startswith("triton"),))

    bf16_fields = [f for f in ms_fields if "quant" not in f]
    quant_fields = [f for f in ms_fields if "quant" in f]

    def _collect(fields):
        out = []
        for field in fields:
            vals = [float(getattr(r, field) or "nan") for r in ok]
            if any(not math.isnan(v) for v in vals):
                out.append((_field_to_label(field), vals))
        return out

    return _collect(bf16_fields) + _collect(quant_fields)


def _save_fig(fig, stem: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext, kwargs in [
        (".png", dict(dpi=600, bbox_inches="tight", pad_inches=0.1)),
        (".pdf", dict(bbox_inches="tight", pad_inches=0.1)),
        (".svg", dict(bbox_inches="tight", pad_inches=0.1)),
    ]:
        path = output_dir / f"{stem}{ext}"
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def _plot_line(
    ok: list[BenchmarkResult],
    active: list[tuple[str, list[float]]],
    phase: str,
    output_dir: Path,
) -> list[Path]:
    seqlens = [r.config.seqlen_k for r in ok]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("white")
    x = np.array(seqlens, dtype=float)

    for idx, (label, vals) in enumerate(active):
        ci = f"C{idx}"
        ax.plot(x, vals, label=label, color=ci, linewidth=2, linestyle="-")
        ax.scatter(
            x,
            vals,
            s=420,
            color=ci,
            alpha=0.24,
            edgecolor=ci,
            linewidth=1.6,
            zorder=3,
        )
        ax.scatter(
            x,
            vals,
            s=140,
            color=ci,
            alpha=1.0,
            edgecolor="#4a4a4a",
            linewidth=0.9,
            zorder=4,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator(seqlens))
    ax.set_xticks(seqlens)
    ax.set_xticklabels(
        [str(v) for v in seqlens],
        rotation=30,
        ha="right",
        fontsize=12,
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(
        f"Attention {phase} latency with head dim {ok[0].config.head_dim}",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Latency (ms)", fontsize=14)
    ax.set_xlabel("Sequence Length", fontsize=14)
    ax.grid(False, which="both")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.legend(
        loc="upper left",
        fontsize=12,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#cccccc",
        facecolor="white",
    )
    fig.tight_layout()

    return _save_fig(fig, f"latency_{phase}", output_dir)


def _plot_bar(
    ok: list[BenchmarkResult],
    active: list[tuple[str, list[float]]],
    phase: str,
    output_dir: Path,
) -> list[Path]:
    seqlens = [r.config.seqlen_k for r in ok]
    n_groups = len(seqlens)
    n_methods = len(active)

    # Bar geometry — thin bars, tight intra-group gap, wider inter-group gap
    bar_width = 0.12
    intra_gap = 0.02
    group_width = n_methods * bar_width + (n_methods - 1) * intra_gap
    inter_gap = group_width * 0.6

    # X positions for each group center
    group_centers = np.zeros(n_groups)
    for i in range(1, n_groups):
        group_centers[i] = group_centers[i - 1] + group_width + inter_gap

    # Dark edge colors — one per method, darkened from the default cycle
    edge_colors = [
        "#1f3a5f",
        "#2f5a3a",
        "#5a2f2f",
        "#4a2f6e",
        "#6d4f1f",
        "#3a5a5f",
        "#5f3a1f",
        "#2f4a3a",
        "#5a3f5f",
        "#4f5a2f",
    ]

    fig, ax = plt.subplots(figsize=(max(10.5, n_groups * 1.8), 4.5))

    for j, (label, vals) in enumerate(active):
        ci = f"C{j}"
        offset = -group_width / 2 + j * (bar_width + intra_gap) + bar_width / 2
        positions = group_centers + offset
        ax.bar(
            positions,
            vals,
            bar_width,
            label=label,
            color=ci,
            edgecolor=edge_colors[j % len(edge_colors)],
            linewidth=1.0,
            hatch=_HATCHES[j % len(_HATCHES)],
            alpha=0.9,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(v) for v in seqlens], rotation=30, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_yscale("log")
    ax.set_title(
        f"Attention {phase} latency with head dim {ok[0].config.head_dim}",
        fontsize=16,
        pad=8,
    )
    ax.set_ylabel("Latency (ms)", fontsize=14)
    ax.set_xlabel("Sequence Length", fontsize=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.legend(
        loc="upper left",
        ncols=2,
        frameon=True,
        framealpha=0.0,
        fontsize=11,
    )
    fig.tight_layout()

    return _save_fig(fig, f"latency_{phase}_bar", output_dir)


def plot_benchmark_results(
    results: List[BenchmarkResult],
    phase: str,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Plot latency charts from benchmark results."""
    if not _HAS_MPL:
        print("[benchmark_plot] WARNING: matplotlib not installed, skipping.")
        return []

    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR

    ok = sorted(
        [r for r in results if r.error_message is None],
        key=lambda r: r.config.seqlen_k,
    )
    if not ok:
        print("[benchmark_plot] No successful results to plot.")
        return []

    active = _discover_active_series(ok)

    saved = _plot_line(ok, active, phase, output_dir)
    saved += _plot_bar(ok, active, phase, output_dir)

    stems = ", ".join(sorted({p.stem for p in saved}))
    print(f"[benchmark_plot] Saved to {output_dir} ({stems})")
    return saved
