#!/usr/bin/env python3
"""
Orzeh Technologies Computer Vision Candidate Screening Task
Entry Point CLI.

Usage:
    python run.py --video <path_to_video> --output <output_directory>

Optional arguments:
    --visualize       Generate an annotated MP4 video with detections and state HUD
    --fps             Frame sampling rate (default: 2.0 FPS)
    --device          Compute device ('cpu' or 'cuda', default: auto-detect)
"""

import argparse
import os
import sys
import time
import pandas as pd
import torch

# Add workspace directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import VideoPipeline
from src.schema import validate_csv, ALL_BEHAVIOURS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Computer Vision Behaviour Screening Pipeline on video."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (e.g., videos/002.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save output CSV results (e.g., results/)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, also renders an annotated demonstration video with HUD overlay",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Sample rate in frames per second (default: 2.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on ('cpu' or 'cuda')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]
    output_csv = os.path.join(args.output, f"{video_id}.csv")

    vis_path = None
    if args.visualize:
        vis_path = os.path.join(args.output, f"{video_id}_demo.mp4")

    print(f"============================================================")
    print(f" Orzeh Technologies CV Screening Pipeline")
    print(f"============================================================")
    print(f" Input Video : {video_path}")
    print(f" Video ID    : {video_id}")
    print(f" Output CSV  : {output_csv}")
    print(f" Device      : {args.device}")
    print(f" Sample FPS  : {args.fps}")
    if vis_path:
        print(f" Demo Video  : {vis_path}")
    print(f"============================================================\n")

    t0 = time.time()
    pipeline = VideoPipeline(
        pose_weights="yolo11n-pose.pt",
        sample_fps=args.fps,
        device=args.device,
    )

    def on_progress(cur_t, dur_t):
        pct = (cur_t / max(dur_t, 0.001)) * 100.0
        sys.stdout.write(f"\r Processing: {cur_t:05.1f}s / {dur_t:05.1f}s [{pct:05.1f}%]")
        sys.stdout.flush()

    intervals = pipeline.process_video(
        video_path=video_path,
        video_id=video_id,
        visualize_output_path=vis_path,
        progress_callback=on_progress,
    )
    print()

    # Convert to Table 3 DataFrame
    rows = [it.to_dict() for it in intervals]
    df = pd.DataFrame(rows)

    # Validate output
    last_end = float(df["end_sec"].max()) if len(df) > 0 else 0.0
    valid, errors = validate_csv(df, video_id=video_id, video_duration=last_end)
    if not valid:
        print(f"\n[WARNING] Output validation reported {len(errors)} warnings:")
        for err in errors:
            print(f"  - {err}")
    else:
        print(f"\n[SUCCESS] Strict Table 3 CSV validation PASSED!")

    # Save CSV
    df.to_csv(output_csv, index=False)
    elapsed = time.time() - t0
    print(f"[SUCCESS] Wrote {len(df)} intervals to {output_csv} in {elapsed:.1f}s.")

    # Summary table
    print("\nSummary of Generated Intervals:")
    for b in ALL_BEHAVIOURS:
        sub = df[df["behaviour"] == b]
        states_summary = ", ".join(f"{s} ({len(g)} intervals)" for s, g in sub.groupby("state"))
        print(f"  * {b:<20}: {states_summary}")

    print("\nDone.")


if __name__ == "__main__":
    main()

