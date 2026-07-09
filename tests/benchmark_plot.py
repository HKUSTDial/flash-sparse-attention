from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import List, Optional

import torch


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

_HATCHES = ["///", "\\\\", "||", "..", "++", "OO", "**", "xx"]

_SERIES_COLORS = {
    "FA": "C0",  # blue
    "cuDNN": "C1",  # orange
    "FSA Base": "C2",  # green
    "FSA Flex Window": "C4",  # purple
    "FSA Split Combine": "C5",  # brown
    "FSA Fused Quant": "C6",  # pink
    "FSA Sparse Softmax": "C7",  # gray
    "FSA All": "C3",  # red
}


def _field_to_label(field_name: str) -> str:
    _LABEL_MAP = {
        "fa": "FA",
        "cudnn": "cuDNN",
        "fsa_base": "FSA Base",
        "fsa_window": "FSA Flex Window",
        "fsa_split": "FSA Split Combine",
        "fsa_quant": "FSA Fused Quant",
        "fsa_skip": "FSA Sparse Softmax",
        "fsa_all": "FSA All",
    }
    stem = field_name.removesuffix("_ms")
    if stem in _LABEL_MAP:
        return _LABEL_MAP[stem]
    parts = stem.split("_")
    return " ".join(p.upper() if p in ("fsa",) else p.capitalize() for p in parts)


def _discover_active_series(ok: list):
    result_cls = type(ok[0])
    ms_fields = [
        f.name for f in dataclasses.fields(result_cls) if f.name.endswith("_ms")
    ]

    baseline = [f for f in ms_fields if not f.startswith("fsa")]
    fsa = [f for f in ms_fields if f.startswith("fsa")]

    ordered = baseline + fsa

    def _collect(fields):
        out = []
        for field in fields:
            vals = [float(getattr(r, field) or "nan") for r in ok]
            if any(not math.isnan(v) for v in vals):
                out.append((_field_to_label(field), vals))
        return out

    return _collect(ordered)


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


def _normalize_device_name(device_name: str) -> str:
    return device_name.removeprefix("NVIDIA ").strip()


def _get_device_name() -> str | None:
    if torch.cuda.is_available():
        return _normalize_device_name(
            torch.cuda.get_device_name(torch.cuda.current_device())
        )
    return None


def _title_for(ok, phase: str, device_name: str | None) -> str:
    title = f"Attention {phase} latency"
    if device_name:
        title += f" on {device_name}"
    return title


def _config_info_text(ok) -> str:
    cfg = ok[0].config
    return (
        f"batch={cfg.batch_size}  heads={cfg.num_heads}  "
        f"kv_heads={cfg.num_kv_heads}  head_dim={cfg.head_dim}"
    )


def _plot_line(
    ok,
    active: list[tuple[str, list[float]]],
    phase: str,
    output_dir: Path,
    device_name: str | None,
) -> list[Path]:
    seqlens = [r.config.seqlen_k for r in ok]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("white")
    x = np.array(seqlens, dtype=float)

    # Identify baseline series for speedup annotations
    baseline_vals = None
    for label, vals in active:
        if label == "FA":
            baseline_vals = vals
            break

    for idx, (label, vals) in enumerate(active):
        ci = _SERIES_COLORS.get(label, f"C{idx}")
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

        # Annotate speedup relative to FA on non-FA series
        if baseline_vals is not None and label != "FA":
            for i, (xi, val, base) in enumerate(zip(x, vals, baseline_vals)):
                if math.isnan(val) or math.isnan(base) or val <= 0:
                    continue
                speedup = base / val
                ax.annotate(
                    f"{speedup:.1f}x",
                    xy=(xi, val),
                    xytext=(0, -14),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    color=ci,
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
        _title_for(ok, phase, device_name),
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
    # Show config info to the right of the legend area
    ax.annotate(
        _config_info_text(ok),
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
        color="#555555",
    )
    fig.tight_layout()

    return _save_fig(fig, f"latency_{phase}", output_dir)


def _plot_bar(
    ok,
    active: list[tuple[str, list[float]]],
    phase: str,
    output_dir: Path,
    device_name: str | None,
) -> list[Path]:
    seqlens = [r.config.seqlen_k for r in ok]
    n_groups = len(seqlens)
    n_methods = len(active)

    # Bar geometry
    bar_width = 0.1
    intra_gap = 0.05
    group_width = n_methods * bar_width + (n_methods - 1) * intra_gap
    inter_gap = 0.2

    # X positions for each group center
    group_centers = np.zeros(n_groups)
    for i in range(1, n_groups):
        group_centers[i] = group_centers[i - 1] + group_width + inter_gap

    # Dark edge colors
    _EDGE_COLORS = {
        "FA": "#0e3b5a",  # dark blue (C0)
        "cuDNN": "#b35608",  # dark orange (C1)
        "FSA Base": "#1a6b1a",  # dark green (C2)
        "FSA Flex Window": "#5c3d7a",  # dark purple (C4)
        "FSA Split Combine": "#5a3730",  # dark brown (C5)
        "FSA Fused Quant": "#a34d8a",  # dark pink (C6)
        "FSA Sparse Softmax": "#4a4a4a",  # dark gray (C7)
        "FSA All": "#8b1a1a",  # dark red (C3)
    }

    fig, ax = plt.subplots(figsize=(max(10.5, n_groups * 1.8), 4.5))

    # Identify baseline series for speedup annotations
    baseline_vals = None
    for label, vals in active:
        if label == "FA":
            baseline_vals = vals
            break

    for j, (label, vals) in enumerate(active):
        ci = _SERIES_COLORS.get(label, f"C{j}")
        offset = -group_width / 2 + j * (bar_width + intra_gap) + bar_width / 2
        positions = group_centers + offset
        ax.bar(
            positions,
            vals,
            bar_width,
            label=label,
            color=ci,
            edgecolor=_EDGE_COLORS.get(label, "#333333"),
            linewidth=1.0,
            hatch=_HATCHES[j % len(_HATCHES)],
            alpha=0.9,
        )

        # Annotate speedup relative to FA on non-FA bars
        if baseline_vals is not None and label != "FA":
            for i, (pos, val, base) in enumerate(zip(positions, vals, baseline_vals)):
                if math.isnan(val) or math.isnan(base) or val <= 0:
                    continue
                speedup = base / val
                ax.annotate(
                    f"{speedup:.1f}x",
                    xy=(pos, val),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    fontweight="bold",
                    color=ci,
                )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(v) for v in seqlens], rotation=30, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_yscale("log")
    ax.set_title(
        _title_for(ok, phase, device_name),
        fontsize=16,
        pad=8,
    )
    ax.set_ylabel("Latency (ms)", fontsize=14)
    ax.set_xlabel("Sequence Length", fontsize=14)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.legend(
        loc="upper left",
        ncols=1,
        frameon=True,
        framealpha=0.0,
        fontsize=10,
    )
    ax.annotate(
        _config_info_text(ok),
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
        color="#000000",
    )
    fig.tight_layout()

    return _save_fig(fig, f"latency_{phase}_bar", output_dir)


def plot_benchmark_results(
    results,
    phase: str,
    output_dir: Optional[Path] = None,
    device_name: Optional[str] = None,
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
    if device_name is None:
        device_name = _get_device_name()

    saved = _plot_line(ok, active, phase, output_dir, device_name)
    saved += _plot_bar(ok, active, phase, output_dir, device_name)

    stems = ", ".join(sorted({p.stem for p in saved}))
    print(f"[benchmark_plot] Saved to {output_dir} ({stems})")
    return saved
