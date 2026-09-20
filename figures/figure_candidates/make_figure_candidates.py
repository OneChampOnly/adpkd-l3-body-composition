#!/usr/bin/env python3
"""Render Figure 3 / Figure 4 candidates using the published figure code.

Selection reproduces how the current figures were chosen: total absolute CSA error summed
over the four compartments, restricted to cases whose section was localized exactly. On the
2026-09-19 run the published Figure 4 case ranks 2nd worst and the published Figure 3 case
ranks 234th of 258 (i.e. among the lowest errors) under this criterion.

Restricting to exactly localized cases matters. The five worst cases overall are all L3
segmentation failures that selected a section 3-17 slices away, so their overlays would show
a different anatomical level rather than a hard segmentation problem. Those are listed in
localization_failures.csv for reference but are not rendered as figure candidates.

Panel rendering is delegated to make_inferior_segmentation_maps.create_case, so the output
is byte-identical in style to the published figures.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper_data/segmentation_map"))
from make_inferior_segmentation_maps import create_case  # noqa: E402

TISSUES = ["sma", "vfat", "sfat", "imat"]
N_CANDIDATES = 5
# Must match make_inferior_segmentation_maps.parse_args defaults, otherwise the
# candidates are windowed differently from the published figures.
ALPHA, WINDOW_MIN, WINDOW_MAX, DPI = 1.0, -200.0, 300.0, 300


def load_inferior() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "results/evaluation/per_case_end_to_end.csv", dtype={"case_id": str})
    frame = frame[frame.branch == "inferior"].copy()
    frame["mean_dsc"] = frame[[f"dsc_{t}" for t in TISSUES]].mean(axis=1)
    frame["total_abs_error_cm2"] = frame[[f"abs_error_{t}_cm2" for t in TISSUES]].sum(axis=1)
    return frame


def caption(row: pd.Series, kind: str) -> str:
    """The DSC sentence the figure legend needs, in the published wording."""
    dscs = ", ".join(f"{row[f'dsc_{t}']:.2f}" for t in TISSUES)
    lead = ("Representative case with accurate fully automated segmentation."
            if kind == "best" else
            "Representative case with the largest fully automated measurement error.")
    return (f"{lead} Axial CT images at the L3 inferior endplate level in a patient with ADPKD "
            f"show the automated segmentation results for SMA (A, yellow), VFAT (B, green), "
            f"SFAT (C, blue), and IMAT (D, red). E. The combined overlay of all segmentation "
            f"areas. The DSCs were {dscs} for SMA, VFAT, SFAT, and IMAT, respectively.")


def main() -> None:
    inferior = load_inferior()
    localized = inferior[inferior.abs_slice_error == 0]
    print(f"{len(inferior)} cases; {len(localized)} localized exactly")

    pools = {
        "best": localized.nsmallest(N_CANDIDATES, "total_abs_error_cm2"),
        "worst": localized.nlargest(N_CANDIDATES, "total_abs_error_cm2"),
    }

    rows: list[dict[str, object]] = []
    captions: list[dict[str, str]] = []
    for kind, pool in pools.items():
        for rank, (_, case) in enumerate(pool.iterrows(), start=1):
            case_id, fold = case.case_id, int(case.fold)
            tag = f"{kind}{rank:02d}"  # create_case appends the case id to the folder name
            print(f"[{kind} {rank}/{N_CANDIDATES}] {case_id} fold {fold} "
                  f"totErr={case.total_abs_error_cm2:.2f} cm2 meanDSC={case.mean_dsc:.3f}", flush=True)
            panels = create_case(tag, fold, case_id, OUT / kind, ALPHA, WINDOW_MIN, WINDOW_MAX, DPI)
            for panel in panels:
                panel.update({
                    "candidate_pool": kind, "candidate_rank": rank,
                    "total_abs_error_cm2": float(case.total_abs_error_cm2),
                    "mean_dsc": float(case.mean_dsc),
                    **{f"dsc_{t}": float(case[f"dsc_{t}"]) for t in TISSUES},
                    **{f"abs_error_{t}_cm2": float(case[f"abs_error_{t}_cm2"]) for t in TISSUES},
                    "l3_dsc": float(case.l3_dsc),
                    "gt_slice_index": int(case.gt_slice_index),
                    "pred_slice_index": int(case.pred_slice_index),
                })
                rows.append(panel)
            captions.append({
                "candidate_pool": kind, "candidate_rank": rank, "case_id": case_id, "fold": fold,
                "total_abs_error_cm2": f"{case.total_abs_error_cm2:.2f}",
                "mean_dsc": f"{case.mean_dsc:.3f}",
                "caption": caption(case, kind),
            })

    with (OUT / "figure_candidates_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "figure_candidate_captions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(captions[0]))
        writer.writeheader()
        writer.writerows(captions)

    # Worst overall, for the record: these failed localization, not segmentation.
    failures = inferior[inferior.abs_slice_error > 0].nsmallest(10, "mean_dsc")[
        ["case_id", "fold", "l3_dsc", "slice_error", "mean_dsc", "total_abs_error_cm2"]
        + [f"dsc_{t}" for t in TISSUES]]
    failures.to_csv(OUT / "localization_failures.csv", index=False)

    lines = ["# Figure 3 / Figure 4 candidates", "",
             "Selected by total absolute CSA error over the four compartments, among cases whose",
             "section was localized exactly (n=%d of %d)." % (len(localized), len(inferior)), ""]
    for kind, title in (("best", "Best measurement (Figure 3 candidates)"),
                        ("worst", "Largest measurement error (Figure 4 candidates)")):
        lines += [f"## {title}", "",
                  "| # | Case | Fold | Total error, cm² | Mean DSC | SMA | VFAT | SFAT | IMAT |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for rank, (_, c) in enumerate(pools[kind].iterrows(), start=1):
            lines.append(f"| {rank} | {c.case_id} | {int(c.fold)} | {c.total_abs_error_cm2:.2f} | "
                         f"{c.mean_dsc:.3f} | " + " | ".join(f"{c[f'dsc_{t}']:.3f}" for t in TISSUES) + " |")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
