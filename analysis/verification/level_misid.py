#!/usr/bin/env python3
"""Level-misidentification adjudication and the tables that depend on it.

SD is the sample standard deviation (ddof=1), matching every other SD in the paper.

Criterion (fixed before counting, per the reviewer's request)
------------------------------------------------------------
A case is a LEVEL MISIDENTIFICATION when the localisation error exceeds one
L3 vertebral body height, measured from this cohort's own ground truth:
the inferior and middle landmarks are the inferior endplate and the mid-body,
so twice their mean separation (2 x 10.3 mm) is the body height, 20.6 mm.
An error larger than that cannot be a within-L3 offset of the reference
section; the predicted section sits on an adjacent vertebra.

    THRESHOLD = 20.0 mm

The cohort's errors are bimodal with an empty band between ~16 and ~25 mm,
so the count is insensitive to the exact cut: identical for 20-22 mm and
differing by at most one case over 15-25 mm (see threshold_sensitivity.csv).
"""
import csv, statistics as st
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
THRESHOLD = 20.0
CONDS = [("native_e2e", "Native end-to-end"), ("z5mm", "5 mm resampled"),
         ("z3mm", "3 mm resampled"), ("gtmask", "Manual L3 mask -> detector")]
BRANCHES = ("inferior", "middle")

rows = list(csv.DictReader(open(HERE / "localisation_original_mm_per_case.csv")))
def sub(c, b): return {r["case_id"]: r for r in rows if r["condition"] == c and r["branch"] == b}

def pct(v, q): return float(np.percentile(v, q))

def block(recs):
    """median [IQR], mean +- SD and the within-tolerance proportions."""
    a = [float(r["abs_mm"]) for r in recs]
    s = [float(r["signed_mm"]) for r in recs]
    n = len(a)
    return dict(
        n=n,
        abs_median=st.median(a), abs_q1=pct(a, 25), abs_q3=pct(a, 75),
        abs_mean=st.mean(a), abs_sd=st.stdev(a), abs_max=max(a),
        signed_mean=st.mean(s), signed_sd=st.stdev(s),
        within_1_slice=100 * sum(r["within_1_native_slice"] in ("True", "1") for r in recs) / n,
        within_2_slices=100 * sum(r["within_2_native_slices"] in ("True", "1") for r in recs) / n,
        within_5mm=100 * sum(x <= 5 + 1e-6 for x in a) / n,
        within_10mm=100 * sum(x <= 10 + 1e-6 for x in a) / n,
    )

# ---- 1. threshold sensitivity -------------------------------------------
with open(HERE / "threshold_sensitivity.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["condition", "branch"] + [f"T={t}" for t in (12, 15, 18, 20, 22, 25, 28)])
    for c, _ in CONDS:
        for b in BRANCHES:
            a = [float(r["abs_mm"]) for r in sub(c, b).values()]
            w.writerow([c, b] + [sum(1 for x in a if x > t) for t in (12, 15, 18, 20, 22, 25, 28)])

# ---- 2. misidentification inventory -------------------------------------
inv = []
for c, _ in CONDS:
    for b in BRANCHES:
        for cid, r in sorted(sub(c, b).items()):
            e = float(r["signed_mm"])
            if abs(e) > THRESHOLD:
                inv.append(dict(condition=c, branch=b, case_id=cid, signed_mm=e,
                                abs_mm=abs(e), direction="cranial" if e > 0 else "caudal",
                                native_spacing=r["native_spacing"]))
with open(HERE / "level_misid_cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(inv[0])); w.writeheader(); w.writerows(inv)

with open(HERE / "level_misid_counts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "branch", "n_total", "n_misid", "pct_misid", "n_cranial", "n_caudal", "max_abs_mm"])
    for c, _ in CONDS:
        for b in BRANCHES:
            m = [r for r in inv if r["condition"] == c and r["branch"] == b]
            w.writerow([c, b, 300, len(m), round(100 * len(m) / 300, 1),
                        sum(r["direction"] == "cranial" for r in m),
                        sum(r["direction"] == "caudal" for r in m),
                        round(max((r["abs_mm"] for r in m), default=0.0), 1)])

# ---- 3. main + sensitivity tables ---------------------------------------
out = []
for c, label in CONDS:
    for b in BRANCHES:
        recs = list(sub(c, b).values())
        keep = [r for r in recs if abs(float(r["signed_mm"])) <= THRESHOLD]
        out.append(dict(condition=c, label=label, branch=b, population="all_300", **block(recs)))
        out.append(dict(condition=c, label=label, branch=b, population="misid_excluded", **block(keep)))
with open(HERE / "localisation_tables.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

# ---- 4. console report ---------------------------------------------------
print(f"LEVEL MISIDENTIFICATION  (|error| > {THRESHOLD:.0f} mm = one L3 body height)\n")
print(f"{'condition':<20}{'branch':<10}{'n':>4}{'%':>7}{'cranial':>9}{'caudal':>8}{'max mm':>9}")
for c, label in CONDS:
    for b in BRANCHES:
        m = [r for r in inv if r["condition"] == c and r["branch"] == b]
        print(f"{label:<20}{b:<10}{len(m):>4}{100*len(m)/300:>6.1f}%"
              f"{sum(r['direction']=='cranial' for r in m):>9}"
              f"{sum(r['direction']=='caudal' for r in m):>8}"
              f"{max((r['abs_mm'] for r in m), default=0.0):>9.1f}")

print("\n\nCase overlap across conditions (same scan failing in more than one)")
for b in BRANCHES:
    ids = {}
    for r in inv:
        if r["branch"] == b: ids.setdefault(r["case_id"], []).append(r["condition"])
    print(f"  {b}: {len(ids)} distinct scans")
    for cid, cs in sorted(ids.items(), key=lambda kv: -len(kv[1])):
        print(f"     {cid}  {len(cs)}/4 conditions  {','.join(cs)}")

print("\n\nLOCALISATION, all 300 scans   (median [IQR]; mean +- SD)")
hdr = f"{'condition':<20}{'branch':<10}{'median [IQR]':>20}{'mean+-SD':>16}{'<=1sl':>8}{'<=2sl':>8}{'<=5mm':>8}{'<=10mm':>8}"
for pop in ("all_300", "misid_excluded"):
    print(f"\n--- {pop} ---"); print(hdr)
    for r in out:
        if r["population"] != pop: continue
        print(f"{r['label']:<20}{r['branch']:<10}"
              f"{r['abs_median']:>8.1f} [{r['abs_q1']:.1f}-{r['abs_q3']:.1f}]".rjust(20)
              + f"{r['abs_mean']:>8.2f}+-{r['abs_sd']:<6.2f}"
              + f"{r['within_1_slice']:>7.1f}%{r['within_2_slices']:>7.1f}%"
              + f"{r['within_5mm']:>7.1f}%{r['within_10mm']:>7.1f}%")
