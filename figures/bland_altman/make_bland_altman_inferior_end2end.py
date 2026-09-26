#!/usr/bin/env python3
"""Bland-Altman analysis of inferior GT versus full end-to-end predictions."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Vector output: embed TrueType (type 42) rather than type-3 bitmapped glyphs, so the
# text in the PDF stays selectable, searchable and sharp at any zoom level.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

# Base point sizes at FONT_SCALE = 1.0; --font-scale multiplies all of them together
# so the relative hierarchy is preserved.
FONT = {
    "panel_title": 16.0,
    "panel_letter": 22.0,
    "axis_label": 17.0,
    "tick": 15.0,
    "loa_label": 14.0,
    "suptitle": 22.0,
}
FONT_SCALE = 1.0
# "oneline": '+1.96 SD  5.55' on one row, tucked just above/below its line.
# "stacked": caption over value, centred on the line, nudged apart when two lines
#            sit too close to fit. Both are drawn inside the frame.
LOA_STYLE = "stacked"


def fs(key: str) -> float:
    return FONT[key] * FONT_SCALE


ROOT = Path(os.environ.get("ADPKD_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_INPUT = ROOT / "results/evaluation/per_case_end_to_end.csv"
DEFAULT_OUTPUT = ROOT / "paper_data/bland_altman_inferior_end2end"

TISSUES = {
    "sma": {"name": "SMA", "color": "#4056A1", "x_label": "Mean of SMA area (cm²)", "x_limits": (0.0, 250.0), "x_tick_step": 50.0, "y_tick_step": 5.0},
    "sfat": {"name": "SFAT", "color": "#4056A1", "x_label": "Mean of SFAT area (cm²)", "x_limits": (0.0, 400.0), "x_tick_step": 100.0, "y_tick_step": 5.0},
    "vfat": {"name": "VFAT", "color": "#4056A1", "x_label": "Mean of VFAT area (cm²)", "x_limits": (0.0, 400.0), "x_tick_step": 100.0, "y_tick_step": 5.0},
    "imat": {"name": "IMAT", "color": "#4056A1", "x_label": "Mean of IMAT area (cm²)", "x_limits": (0.0, 20.0), "x_tick_step": 2.5, "y_tick_step": 1.0},
}


# Panel order of the combined 2x2 figure. TISSUES keeps its own order so that the
# summary and per-case CSVs are unaffected.
COMBINED_ORDER = ("sma", "vfat", "sfat", "imat")


def finite_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_inferior_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {"fold", "case_id", "branch"}
        for tissue in TISSUES:
            required.update({f"gt_{tissue}_cm2", f"pred_{tissue}_cm2"})
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        rows = [row for row in reader if row["branch"].strip().lower() == "inferior"]

    ids = [row["case_id"].strip() for row in rows]
    if len(rows) != 300 or len(set(ids)) != 300:
        raise RuntimeError(f"Expected 300 unique inferior cases, found rows={len(rows)}, unique={len(set(ids))}")
    return rows


def padded_limits(values: np.ndarray, fraction: float = 0.08) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    pad = max((high - low) * fraction, 0.5) if low != high else max(abs(low) * fraction, 1.0)
    return low - pad, high + pad


def tissue_values(rows: list[dict[str, str]], tissue: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (finite_float(row[f"gt_{tissue}_cm2"]), finite_float(row[f"pred_{tissue}_cm2"]))
        for row in rows
    ]
    if any(not math.isfinite(gt) or not math.isfinite(pred) for gt, pred in pairs):
        raise RuntimeError(f"Non-finite {tissue.upper()} GT/prediction value detected")
    return np.asarray([x[0] for x in pairs]), np.asarray([x[1] for x in pairs])


def calculate_stats(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    differences = pred - gt
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    lower = bias - 1.96 * sd
    upper = bias + 1.96 * sd
    return {
        "n": float(len(differences)),
        "bias": bias,
        "sd": sd,
        "loa_lower": lower,
        "loa_upper": upper,
        "outside_loa": float(np.sum((differences < lower) | (differences > upper))),
    }


def draw_loa_labels(ax: plt.Axes, stats: dict[str, float]) -> None:
    """Label the mean and the limits of agreement inside the frame, right-aligned.

    Values carry the typographic minus so they match the "−1.96 SD" caption.
    """
    entries = [(stats["loa_upper"], "+1.96 SD"), (stats["bias"], "Mean"),
               (stats["loa_lower"], "−1.96 SD")]
    size = fs("loa_label")
    common = {"ha": "right", "fontsize": size, "zorder": 9,
              "bbox": {"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.85}}

    if LOA_STYLE == "oneline":
        # One row per line, tucked clear of it. Half the height of the stacked form,
        # which is what keeps it from colliding where the limits sit close together.
        gap = 2.5 * FONT_SCALE
        for (value, caption), (va, sign) in zip(entries, (("bottom", 1), ("bottom", 1), ("top", -1))):
            text = f"{caption}  {value:.2f}".replace("-", "\u2212")
            ax.annotate(text, xy=(0.985, value), xycoords=ax.get_yaxis_transform(),
                        xytext=(0, sign * gap), textcoords="offset points", va=va, **common)
        return

    # Stacked: centre each two-line block on its line, then push blocks apart if the
    # lines are closer together than the text is tall.
    y0, y1 = ax.get_ylim()
    height_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
    block_pt = 2 * size * 1.2
    min_sep = (block_pt + 2.0) * (y1 - y0) / height_pt   # in data units
    placed = [v for v, _ in entries]                     # upper, mean, lower
    if placed[0] - placed[1] < min_sep:
        shift = (min_sep - (placed[0] - placed[1])) / 2.0
        placed[0] += shift; placed[1] -= shift
    if placed[1] - placed[2] < min_sep:
        placed[2] = placed[1] - min_sep
    half = min_sep / 2.0
    placed = [min(max(v, y0 + half), y1 - half) for v in placed]
    for (value, caption), y in zip(entries, placed):
        text = f"{caption}\n{value:.2f}".replace("-", "\u2212")
        ax.annotate(text, xy=(0.985, y), xycoords=ax.get_yaxis_transform(),
                    va="center", linespacing=1.2, **common)


def draw_panel(ax: plt.Axes, gt: np.ndarray, pred: np.ndarray, tissue: str, panel_title: str | None = None,
               corner_letter: str | None = None) -> dict[str, float]:
    info = TISSUES[tissue]
    means = (gt + pred) / 2.0
    differences = pred - gt
    stats = calculate_stats(gt, pred)

    ax.scatter(
        means,
        differences,
        s=14 * FONT_SCALE**1.5,
        facecolors="white",
        edgecolors=info["color"],
        linewidths=0.6 * FONT_SCALE,
        alpha=0.82,
    )

    x_min, x_max = info["x_limits"]
    y_values = np.concatenate((differences, [stats["loa_lower"], stats["loa_upper"]]))
    y_step = info["y_tick_step"]
    y_min = math.floor(float(np.min(y_values)) / y_step) * y_step
    y_max = math.ceil(float(np.max(y_values)) / y_step) * y_step
    x_pad = 0.025 * (x_max - x_min)
    y_pad = 0.4 * y_step
    x_plot_min, x_plot_max = x_min - x_pad, x_max + x_pad
    line_style = {"color": "#202020", "solid_capstyle": "butt", "dash_capstyle": "butt"}
    ax.axhline(stats["bias"], linewidth=1.05 * FONT_SCALE, **line_style)
    ax.axhline(stats["loa_upper"], linewidth=0.9 * FONT_SCALE, linestyle="--", **line_style)
    ax.axhline(stats["loa_lower"], linewidth=0.9 * FONT_SCALE, linestyle="--", **line_style)
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_xticks(np.arange(x_min, x_max + info["x_tick_step"] / 2.0, info["x_tick_step"]))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_yticks(np.arange(y_min, y_max + y_step / 2.0, y_step))
    if not corner_letter:
        ax.set_title(panel_title or f"Bland–Altman Plot for {info['name']}", fontsize=fs("panel_title"), pad=9)
    ax.set_xlabel(info["x_label"], fontsize=fs("axis_label"))
    ax.set_ylabel("Difference (cm²)", fontsize=fs("axis_label"))
    ax.tick_params(axis="both", labelsize=fs("tick"), direction="in",
                   width=0.8 * FONT_SCALE, length=3.5 * FONT_SCALE)

    draw_loa_labels(ax, stats)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8 * FONT_SCALE)
        ax.spines[side].set_color("#222222")
    return stats


def place_panel_letters(fig, axes, letters=("A", "B", "C", "D")) -> None:
    """Put each panel letter at the outer lower-left corner of its subplot.

    The anchor is the axes' tight bounding box, which includes the tick labels and the
    axis titles, so the letter sits outside the plotting frame at the very corner of the
    drawn panel rather than inside it. Call this after tight_layout().
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for letter, ax in zip(letters, axes):
        box = ax.get_tightbbox(renderer).transformed(inv)
        fig.text(box.x0, box.y0, letter, ha="left", va="bottom",
                 fontsize=fs("panel_letter"), fontweight="bold")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="multiply every text size (and the marker/line weights that "
                             "balance it) by this factor")
    parser.add_argument("--no-pdf", action="store_true", help="skip the vector PDF output")
    parser.add_argument("--loa-style", choices=("oneline", "stacked"), default="stacked",
                        help="layout of the mean / limits-of-agreement labels inside the frame")
    args = parser.parse_args()

    global FONT_SCALE, LOA_STYLE
    FONT_SCALE = args.font_scale
    LOA_STYLE = args.loa_style

    rows = load_inferior_rows(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    per_case_rows: list[dict[str, object]] = []
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for tissue, info in TISSUES.items():
        gt, pred = tissue_values(rows, tissue)
        values[tissue] = (gt, pred)
        fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=200)
        stats = draw_panel(ax, gt, pred, tissue)
        fig.tight_layout()
        png = args.output_dir / f"bland_altman_{tissue}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        if not args.no_pdf:
            fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
        ax.set_title("")
        fig.tight_layout()
        no_title_png = args.output_dir / f"bland_altman_{tissue}_no_title.png"
        fig.savefig(no_title_png, dpi=300, bbox_inches="tight")
        if not args.no_pdf:
            fig.savefig(no_title_png.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

        summary_rows.append({
            "tissue": info["name"],
            "n": int(stats["n"]),
            "mean_difference_cm2": f"{stats['bias']:.6f}",
            "sd_cm2": f"{stats['sd']:.6f}",
            "loa_lower_cm2": f"{stats['loa_lower']:.6f}",
            "loa_upper_cm2": f"{stats['loa_upper']:.6f}",
            "outside_loa_n": int(stats["outside_loa"]),
            "png_path": str(png),
        })

        for row, gt_value, pred_value in zip(rows, gt, pred):
            difference = float(pred_value - gt_value)
            per_case_rows.append({
                "fold": row["fold"],
                "case_id": row["case_id"],
                "tissue": info["name"],
                "gt_cm2": f"{gt_value:.6f}",
                "end_to_end_cm2": f"{pred_value:.6f}",
                "mean_cm2": f"{(gt_value + pred_value) / 2:.6f}",
                "difference_pred_minus_gt_cm2": f"{difference:.6f}",
                "outside_loa": int(difference < stats["loa_lower"] or difference > stats["loa_upper"]),
            })

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), dpi=200)
    for label, (ax, tissue) in zip(("A", "B", "C", "D"), zip(axes.flat, COMBINED_ORDER)):
        gt, pred = values[tissue]
        draw_panel(ax, gt, pred, tissue, corner_letter=label)  # letter drawn after layout
    main_title = fig.suptitle("Inferior GT vs End-to-End Prediction: Bland–Altman Analysis",
                              fontsize=fs("suptitle"), y=1.01)
    fig.tight_layout(w_pad=5.0, h_pad=3.0)
    place_panel_letters(fig, list(axes.flat))
    combined = args.output_dir / "bland_altman_combined.png"
    fig.savefig(combined, dpi=300, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(combined.with_suffix(".pdf"), bbox_inches="tight")
    main_title.remove()
    fig._suptitle = None
    # Journal variant: the overall title belongs in the legend, but the A-D panel
    # labels must stay, so this is saved before the panel titles are cleared.
    fig.tight_layout()
    panels_only = args.output_dir / "bland_altman_combined_panels_only.png"
    fig.savefig(panels_only, dpi=300, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(panels_only.with_suffix(".pdf"), bbox_inches="tight")
    fig.tight_layout()
    combined_no_title = args.output_dir / "bland_altman_combined_no_title.png"
    fig.savefig(combined_no_title, dpi=300, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(combined_no_title.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    write_csv(
        args.output_dir / "bland_altman_summary.csv",
        ["tissue", "n", "mean_difference_cm2", "sd_cm2", "loa_lower_cm2", "loa_upper_cm2", "outside_loa_n", "png_path"],
        summary_rows,
    )
    write_csv(
        args.output_dir / "bland_altman_per_case.csv",
        ["fold", "case_id", "tissue", "gt_cm2", "end_to_end_cm2", "mean_cm2", "difference_pred_minus_gt_cm2", "outside_loa"],
        per_case_rows,
    )
    print(f"[OK] inferior cases: {len(rows)}")
    print(f"[OK] combined figure: {combined}")
    print(f"[OK] combined figure without titles: {combined_no_title}")
    print(f"[OK] summary: {args.output_dir / 'bland_altman_summary.csv'}")


if __name__ == "__main__":
    main()
