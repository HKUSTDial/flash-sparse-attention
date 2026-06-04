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


def _field_to_label(field_name: str) -> str:
    """``triton_dense_ms`` → ``Triton Dense``, ``fa_dense_ms`` → ``FA Dense``."""
    parts = field_name.removesuffix("_ms").split("_")
    return " ".join(
        p.upper() if p in ("fa", "cudnn") else p.capitalize() for p in parts
    )


def plot_benchmark_results(
    results: List[BenchmarkResult],
    phase: str,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Plot latency line chart from benchmark results.
    """
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

    ms_fields = [
        f.name for f in dataclasses.fields(BenchmarkResult) if f.name.endswith("_ms")
    ]
    ms_fields.sort(key=lambda n: (n.startswith("triton"),))

    bf16_fields = [f for f in ms_fields if "quant" not in f]
    quant_fields = [f for f in ms_fields if "quant" in f]

    # Extract values, skip series that are all-nan
    seqlens = [r.config.seqlen_k for r in ok]

    def _collect(fields):
        """
        Return [(label, [values...])] for fields that have at least one real value.
        """
        out = []
        for field in fields:
            vals = [float(getattr(r, field) or "nan") for r in ok]
            if any(not math.isnan(v) for v in vals):
                out.append((_field_to_label(field), vals))
        return out

    active_bf16 = _collect(bf16_fields)
    active_quant = _collect(quant_fields)

    active = active_bf16 + active_quant

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("white")
    x = np.array(seqlens, dtype=float)

    for idx, (label, vals) in enumerate(active):
        ci = f"C{idx}"
        ax.plot(x, vals, label=label, color=ci, linewidth=2, linestyle="-")
        ax.scatter(
            x, vals, s=420, color=ci, alpha=0.24, edgecolor=ci, linewidth=1.6, zorder=3
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

    # Axes
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator(seqlens))
    ax.set_xticks(seqlens)
    ax.set_xticklabels([str(v) for v in seqlens], rotation=30, ha="right", fontsize=12)
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

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"latency_{phase}"
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

    print(f"[benchmark_plot] Saved to {output_dir} ({stem})")
    return saved
