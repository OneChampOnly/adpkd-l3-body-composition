#!/usr/bin/env python3
"""Combine five folds and summarize inferior/middle errors in slices and mm."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(rows: list[dict[str, str]], scope: str, fold: str, branch: str) -> dict[str, object]:
    signed_slices = [float(row["signed_slice_error"]) for row in rows]
    absolute_slices = [float(row["absolute_slice_error"]) for row in rows]
    signed_mm = [float(row["signed_error_mm"]) for row in rows]
    absolute_mm = [float(row["absolute_error_mm"]) for row in rows]
    n = len(rows)
    if not n:
        raise RuntimeError(f"No rows for {scope}/{fold}/{branch}")

    def mean_sd(values: list[float]) -> tuple[float, float]:
        return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0

    ss_m, ss_sd = mean_sd(signed_slices)
    as_m, as_sd = mean_sd(absolute_slices)
    sm_m, sm_sd = mean_sd(signed_mm)
    am_m, am_sd = mean_sd(absolute_mm)
    q1, median, q3 = np.percentile(absolute_mm, [25, 50, 75], method="linear")
    result: dict[str, object] = {
        "scope": scope, "fold": fold, "branch": branch, "n": n,
        "signed_slice_error_mean": ss_m, "signed_slice_error_sd": ss_sd,
        "absolute_slice_error_mean": as_m, "absolute_slice_error_sd": as_sd,
        "signed_error_mm_mean": sm_m, "signed_error_mm_sd": sm_sd,
        "absolute_error_mm_mean": am_m, "absolute_error_mm_sd": am_sd,
        "absolute_error_mm_median": median, "absolute_error_mm_q1": q1, "absolute_error_mm_q3": q3,
    }
    for field in ("exact_slice", "within_1_slice", "within_2_slices", "within_5mm", "within_10mm"):
        count = sum(int(row[field]) for row in rows)
        result[f"{field}_n"] = count
        result[f"{field}_percent"] = count / n * 100.0
    return result


def paper_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "Target": row["branch"].capitalize(),
        "Cases": row["n"],
        "Signed error (slices)": f"{row['signed_slice_error_mean']:.2f} ± {row['signed_slice_error_sd']:.2f}",
        "Absolute error (slices)": f"{row['absolute_slice_error_mean']:.2f} ± {row['absolute_slice_error_sd']:.2f}",
        "Signed error (mm)": f"{row['signed_error_mm_mean']:.2f} ± {row['signed_error_mm_sd']:.2f}",
        "Absolute error (mm)": f"{row['absolute_error_mm_mean']:.2f} ± {row['absolute_error_mm_sd']:.2f}",
        "Median absolute error (mm)": f"{row['absolute_error_mm_median']:.2f} [{row['absolute_error_mm_q1']:.2f}–{row['absolute_error_mm_q3']:.2f}]",
        "Exact slice": f"{row['exact_slice_n']} ({row['exact_slice_percent']:.1f}%)",
        "Within ±1 slice / 5 mm": f"{row['within_1_slice_n']} ({row['within_1_slice_percent']:.1f}%)",
        "Within ±2 slices / 10 mm": f"{row['within_2_slices_n']} ({row['within_2_slices_percent']:.1f}%)",
    }


def main() -> None:
    rows: list[dict[str, str]] = []
    for fold in range(5):
        path = ROOT / f"results/fold_{fold}/detection_per_case.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_rows(path))
    if len(rows) != 600:
        raise RuntimeError(f"Expected 600 branch rows, found {len(rows)}")
    if len({(row["case_id"], row["branch"]) for row in rows}) != 600:
        raise RuntimeError("Duplicate or missing case/branch pairs")

    output = ROOT / "results/summary"
    write_rows(output / "per_case_z3mm_detection.csv", rows)
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        for branch in ("inferior", "middle"):
            selected = [row for row in rows if int(row["fold"]) == fold and row["branch"] == branch]
            summaries.append(stats(selected, "fold", str(fold), branch))
    overall = []
    for branch in ("inferior", "middle"):
        selected = [row for row in rows if row["branch"] == branch]
        result = stats(selected, "overall", "all", branch)
        summaries.append(result)
        overall.append(result)
    write_rows(output / "summary_by_fold_and_overall.csv", summaries)
    write_rows(output / "overall_paper_table.csv", [paper_row(row) for row in overall])
    payload = {"target_z_spacing_mm": 3.0, "total_cases": 300, "overall": overall}
    (output / "overall_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
