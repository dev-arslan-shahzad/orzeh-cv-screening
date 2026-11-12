"""
Coverage Evaluator and Report Generator.
Computes exact Time Coverage per behaviour across evaluation videos
excluding 'not_visible' periods per Section 4 of the Orzeh Screening brief.
"""

import os
import glob
import pandas as pd
import numpy as np

# Ground truth intervals verified from raw video inspection
GROUND_TRUTH = {
    "002": {
        "duration": 66.80,
        "hand_covering": [
            (0.0, 1.2, "not_visible"), (1.2, 20.0, "bare_hands"), (20.0, 46.5, "gloves"),
            (46.5, 50.5, "bare_hands"), (50.5, 66.0, "gloves"), (66.0, 66.80, "not_visible")
        ],
        "shirt_sleeve_length": [
            (0.0, 1.2, "not_visible"), (1.2, 6.0, "half_sleeve"), (6.0, 11.5, "not_visible"),
            (11.5, 23.5, "half_sleeve"), (23.5, 27.0, "not_visible"), (27.0, 66.80, "half_sleeve")
        ],
        "head_covering": [
            (0.0, 1.2, "not_visible"), (1.2, 6.0, "bare_head"), (6.0, 11.5, "not_visible"),
            (11.5, 23.5, "bare_head"), (23.5, 52.5, "not_visible"), (52.5, 65.5, "bare_head"),
            (65.5, 66.80, "not_visible")
        ],
        "hand_washing": [
            (0.0, 1.2, "not_visible"), (1.2, 66.0, "not_washing"), (66.0, 66.80, "not_visible")
        ],
    },
    "003": {
        "duration": 158.22,
        "hand_covering": [(0.0, 158.22, "bare_hands")],
        "shirt_sleeve_length": [(0.0, 158.22, "half_sleeve")],
        "head_covering": [
            (0.0, 59.0, "cap"), (59.0, 79.0, "bare_head"), (79.0, 158.22, "cap")
        ],
        "hand_washing": [(0.0, 158.22, "not_washing")],
    },
    "004": {
        "duration": 230.20,
        "hand_covering": [(0.0, 2.0, "not_visible"), (2.0, 230.20, "gloves")],
        "shirt_sleeve_length": [(0.0, 230.20, "full_sleeve")],
        "head_covering": [(0.0, 230.20, "not_visible")],
        "hand_washing": [
            (0.0, 31.5, "not_washing"), (31.5, 34.0, "washing"), (34.0, 230.20, "not_washing")
        ],
    },
}


def state_at(intervals, t: float) -> str:
    for s, e, st in intervals:
        if s <= t < e:
            return st
    return intervals[-1][2] if intervals else "not_visible"


def compute_time_coverage(gt_segs, pred_segs, duration: float, dt: float = 0.05) -> float:
    total_valid_time = 0.0
    matched_time = 0.0

    t = dt / 2.0
    while t < duration:
        gt_state = state_at(gt_segs, t)
        if gt_state != "not_visible":
            total_valid_time += dt
            pred_state = state_at(pred_segs, t)
            if pred_state == gt_state:
                matched_time += dt
        t += dt

    if total_valid_time <= 0:
        return 100.0
    return (matched_time / total_valid_time) * 100.0


def evaluate_results(results_dir: str = "results"):
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_dir}!")
        return

    print("=" * 80)
    print(" ORZEH TECHNOLOGIES CANDIDATE SCREENING - TIME COVERAGE EVALUATION")
    print(" (Pass Requirement: >= 50.0% Time Coverage per behaviour, excluding not_visible)")
    print("=" * 80)

    rows = []
    for csv_path in sorted(csv_files):
        vid_id = os.path.splitext(os.path.basename(csv_path))[0]
        if vid_id not in GROUND_TRUTH:
            continue

        gt_info = GROUND_TRUTH[vid_id]
        dur = gt_info["duration"]
        df = pd.read_csv(csv_path)

        for b in ["hand_covering", "shirt_sleeve_length", "head_covering", "hand_washing"]:
            sub = df[df["behaviour"] == b]
            pred_segs = [
                (float(r["start_sec"]), float(r["end_sec"]), str(r["state"]))
                for _, r in sub.iterrows()
            ]
            gt_segs = gt_info[b]

            cov = compute_time_coverage(gt_segs, pred_segs, dur)
            status = "PASS" if cov >= 50.0 else "FAIL"

            rows.append({
                "Video ID": vid_id,
                "Behaviour": b,
                "Coverage (%)": f"{cov:.1f}%",
                "Threshold": ">= 50.0%",
                "Result": status
            })

    report_df = pd.DataFrame(rows)
    print(report_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    evaluate_results()

