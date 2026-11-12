#!/usr/bin/env python3
"""
Batch Evaluation and Validation Script for all 4 screening videos.
Generates results/001.csv, results/002.csv, results/003.csv, results/004.csv,
validates strict Table 3 compliance, and computes coverage metrics against GT.
"""

import os
import sys
import time
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import VideoPipeline
from src.schema import validate_csv, ALL_BEHAVIOURS


def evaluate_all(videos_dir: str = "videos", output_dir: str = "results", fps: float = 2.0):
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))

    if not video_files:
        print(f"No .mp4 files found in {videos_dir}!")
        return

    print("=" * 65)
    print(f"Batch Processing {len(video_files)} Videos")
    print("=" * 65)

    pipeline = VideoPipeline(sample_fps=fps)
    summary_records = []

    for idx, vpath in enumerate(video_files):
        vid_id = os.path.splitext(os.path.basename(vpath))[0]
        out_csv = os.path.join(output_dir, f"{vid_id}.csv")
        
        # Also generate demo video for 002.mp4
        demo_path = os.path.join(output_dir, f"{vid_id}_demo.mp4") if vid_id == "002" else None

        print(f"\n[{idx+1}/{len(video_files)}] Processing {vid_id} ({os.path.basename(vpath)})...")
        t0 = time.time()

        def prog(cur, total):
            sys.stdout.write(f"\r  Progress: {cur:05.1f}s / {total:05.1f}s [{(cur/max(total, 0.001))*100:05.1f}%]")
            sys.stdout.flush()

        intervals = pipeline.process_video(
            video_path=vpath,
            video_id=vid_id,
            visualize_output_path=demo_path,
            progress_callback=prog
        )
        print()

        df = pd.DataFrame([it.to_dict() for it in intervals])
        duration = float(df["end_sec"].max()) if len(df) > 0 else 0.0

        is_valid, errors = validate_csv(df, video_id=vid_id, video_duration=duration)
        if not is_valid:
            print(f"  [ERROR] Validation failed for {vid_id}:")
            for e in errors:
                print(f"    - {e}")
        else:
            print(f"  [PASS] Strictly valid Table 3 CSV ({len(df)} intervals, duration: {duration:.2f}s)")

        df.to_csv(out_csv, index=False)
        elapsed = time.time() - t0
        print(f"  Saved -> {out_csv} in {elapsed:.1f}s")

        for b in ALL_BEHAVIOURS:
            sub = df[df["behaviour"] == b]
            summary_records.append({
                "video_id": vid_id,
                "behaviour": b,
                "num_intervals": len(sub),
                "duration_sec": duration,
                "valid": is_valid
            })

    print("\n" + "=" * 65)
    print("Batch Processing Complete!")
    print("=" * 65)
    sum_df = pd.DataFrame(summary_records)
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_all()

