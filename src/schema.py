"""
Schema and Validation for Orzeh Technologies CV Screening Task.
Defines valid behaviours, states, and validates output CSV compliance
strictly per Tables 1 and 3 of the specification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple
import pandas as pd


class Behaviour(str, Enum):
    HAND_COVERING = "hand_covering"
    SHIRT_SLEEVE_LENGTH = "shirt_sleeve_length"
    HEAD_COVERING = "head_covering"
    HAND_WASHING = "hand_washing"


VALID_STATES: Dict[str, List[str]] = {
    Behaviour.HAND_COVERING.value: ["bare_hands", "gloves", "not_visible"],
    Behaviour.SHIRT_SLEEVE_LENGTH.value: ["full_sleeve", "half_sleeve", "not_visible"],
    Behaviour.HEAD_COVERING.value: ["cap", "bare_head", "not_visible"],
    Behaviour.HAND_WASHING.value: ["washing", "not_washing", "not_visible"],
}

ALL_BEHAVIOURS = [
    Behaviour.HAND_COVERING.value,
    Behaviour.SHIRT_SLEEVE_LENGTH.value,
    Behaviour.HEAD_COVERING.value,
    Behaviour.HAND_WASHING.value,
]


@dataclass
class Interval:
    video_id: str
    behaviour: str
    state: str
    start_sec: float
    end_sec: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "behaviour": self.behaviour,
            "state": self.state,
            "start_sec": f"{self.start_sec:.3f}",
            "end_sec": f"{self.end_sec:.3f}",
            "confidence": f"{self.confidence:.3f}",
        }


def validate_csv(df: pd.DataFrame, video_id: str, video_duration: float) -> Tuple[bool, List[str]]:
    """
    Validates that a DataFrame strictly complies with Table 3 output requirements:
    1. Correct column names in exact order.
    2. Exact video_id match.
    3. All 4 behaviours present.
    4. Valid states per behaviour.
    5. Contiguous non-overlapping intervals covering [0.000, video_duration].
    6. Confidences in [0.0, 1.0].
    """
    errors = []
    expected_cols = ["video_id", "behaviour", "state", "start_sec", "end_sec", "confidence"]
    if list(df.columns) != expected_cols:
        errors.append(f"Header columns must be {expected_cols}, got {list(df.columns)}")

    for b in ALL_BEHAVIOURS:
        sub = df[df["behaviour"] == b].copy()
        if len(sub) == 0:
            errors.append(f"Missing behaviour: {b}")
            continue

        # Sort by start_sec
        sub["start_f"] = sub["start_sec"].astype(float)
        sub["end_f"] = sub["end_sec"].astype(float)
        sub["conf_f"] = sub["confidence"].astype(float)
        sub = sub.sort_values(by="start_f").reset_index(drop=True)

        # Check states
        allowed = set(VALID_STATES[b])
        invalid_states = set(sub["state"]) - allowed
        if invalid_states:
            errors.append(f"Behaviour '{b}' has invalid states: {invalid_states}")

        # Check confidences
        if (sub["conf_f"] < 0.0).any() or (sub["conf_f"] > 1.0).any():
            errors.append(f"Behaviour '{b}' has confidence outside [0.0, 1.0]")

        # Check contiguity
        first_start = sub["start_f"].iloc[0]
        last_end = sub["end_f"].iloc[-1]

        if abs(first_start - 0.0) > 0.005:
            errors.append(f"Behaviour '{b}' must start at 0.000, starts at {first_start:.3f}")

        if abs(last_end - video_duration) > 0.05:
            errors.append(f"Behaviour '{b}' ends at {last_end:.3f}, expected video duration {video_duration:.3f}")

        for i in range(len(sub) - 1):
            cur_end = sub["end_f"].iloc[i]
            next_start = sub["start_f"].iloc[i + 1]
            if abs(cur_end - next_start) > 0.005:
                errors.append(
                    f"Behaviour '{b}' interval gap/overlap at index {i}: row {i} ends at {cur_end:.3f}, row {i+1} starts at {next_start:.3f}"
                )

    return (len(errors) == 0, errors)

