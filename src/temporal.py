"""
Temporal smoothing, hysteresis (Schmitt trigger), and contiguous interval generation.
Converts discrete per-frame detections into stable continuous time intervals
strictly compliant with Table 3 of the Orzeh Technologies screening task.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from src.schema import Interval, Behaviour, ALL_BEHAVIOURS


def smooth_scores_exponential(scores: List[float], alpha: float = 0.4) -> List[float]:
    """Applies exponential moving average to continuous scores."""
    if not scores:
        return []
    smoothed = [scores[0]]
    for s in scores[1:]:
        smoothed.append(alpha * s + (1.0 - alpha) * smoothed[-1])
    return smoothed


def schmitt_trigger(
    scores: List[float],
    high_th: float = 0.60,
    low_th: float = 0.40,
    initial_state: bool = False
) -> List[bool]:
    """
    Dual-threshold Schmitt trigger to eliminate boundary jitter.
    Turns True when score >= high_th, turns False when score <= low_th,
    and retains previous state in the hysteresis band [low_th, high_th].
    """
    state = initial_state
    states = []
    for s in scores:
        if s >= high_th:
            state = True
        elif s <= low_th:
            state = False
        states.append(state)
    return states


def filter_short_glitches(states: List[str], min_run_length: int = 3) -> List[str]:
    """Removes isolated spurious glitches that last fewer than min_run_length frames."""
    if len(states) <= min_run_length:
        return states[:]
    
    out = states[:]
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        # If run is very short and surrounded by the same state on both sides, bridge it
        if run_len < min_run_length and i > 0 and j < n:
            prev_val = out[i - 1]
            next_val = out[j]
            if prev_val == next_val:
                for k in range(i, j):
                    out[k] = prev_val
        i = j
    return out


def build_contiguous_intervals(
    timestamps: List[float],
    states: List[str],
    confidences: List[float],
    video_id: str,
    behaviour: str,
    video_duration: float,
    min_segment_sec: float = 0.5
) -> List[Interval]:
    """
    Aggregates per-frame states into contiguous intervals spanning [0.000, video_duration].
    Guarantees:
    - Interval 0 starts at exactly 0.000.
    - Final interval ends at exactly video_duration.
    - No gaps or overlaps between intervals (start_sec[i+1] == end_sec[i]).
    - Confidence is the mean confidence over the interval.
    """
    if not timestamps or not states:
        return [Interval(video_id, behaviour, "not_visible", 0.0, video_duration, 0.900)]

    # Step 1: Group consecutive identical states
    raw_segments: List[Dict[str, Any]] = []
    curr_state = states[0]
    curr_start = 0.0
    curr_confs = [confidences[0]]
    
    for i in range(1, len(states)):
        t_prev = timestamps[i - 1]
        t_curr = timestamps[i]
        midpoint = 0.5 * (t_prev + t_curr)
        
        if states[i] == curr_state:
            curr_confs.append(confidences[i])
        else:
            raw_segments.append({
                "state": curr_state,
                "start": curr_start,
                "end": midpoint,
                "confs": curr_confs
            })
            curr_state = states[i]
            curr_start = midpoint
            curr_confs = [confidences[i]]
            
    raw_segments.append({
        "state": curr_state,
        "start": curr_start,
        "end": video_duration,
        "confs": curr_confs
    })

    # Step 2: Merge micro-segments shorter than min_segment_sec
    # (except for short events like washing where minimum is handled upstream)
    merged_segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        duration = seg["end"] - seg["start"]
        if merged_segments and duration < min_segment_sec and len(raw_segments) > 1:
            # Merge into previous segment
            merged_segments[-1]["end"] = seg["end"]
            merged_segments[-1]["confs"].extend(seg["confs"])
        else:
            merged_segments.append(seg)

    # Step 3: Enforce strict contiguity from 0.000 to video_duration
    if not merged_segments:
        return [Interval(video_id, behaviour, "not_visible", 0.0, video_duration, 0.900)]

    merged_segments[0]["start"] = 0.0
    merged_segments[-1]["end"] = video_duration

    # Ensure adjacent endpoints match exactly
    intervals: List[Interval] = []
    for i in range(len(merged_segments)):
        seg = merged_segments[i]
        start_t = round(seg["start"], 3)
        end_t = round(seg["end"], 3)

        if i > 0:
            # Match previous end exactly
            start_t = intervals[-1].end_sec

        if i == len(merged_segments) - 1:
            end_t = round(video_duration, 3)

        # In case start >= end due to rounding, fix it
        if end_t <= start_t:
            end_t = round(start_t + 0.001, 3)

        mean_conf = float(np.mean(seg["confs"])) if seg["confs"] else 0.850
        mean_conf = max(0.0, min(1.0, mean_conf))

        intervals.append(Interval(
            video_id=video_id,
            behaviour=behaviour,
            state=seg["state"],
            start_sec=start_t,
            end_sec=end_t,
            confidence=round(mean_conf, 3)
        ))

    # Final pass to guarantee last interval ends at video_duration
    if intervals:
        intervals[-1].end_sec = round(video_duration, 3)

    return intervals

