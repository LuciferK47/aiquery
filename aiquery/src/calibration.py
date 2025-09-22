"""
calibration.py
Utilities for enforcing realistic agronomic ranges, validating DataFrames,
and comparing synthetic vs real distributions.
"""
from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


# Calibrated agronomic ranges (can be adjusted via config if needed)
RANGES: Dict[str, Tuple[float, float]] = {
    "soil_ph": (3.5, 9.0),
    "moisture": (0.0, 100.0),
    "nitrogen": (0.0, 100.0),
    "phosphorus": (0.0, 50.0),
    "potassium": (0.0, 100.0),
    "temperature": (-20.0, 60.0),
}


def clamp_value(value: float, min_value: float, max_value: float) -> float:
    """Clamp a numeric value between min_value and max_value."""
    return max(min_value, min(max_value, float(value)))


def clamp_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp known sensor columns in the DataFrame to calibrated ranges.

    Columns not present are ignored.
    """
    clamped = df.copy()
    for col, (mn, mx) in RANGES.items():
        if col in clamped.columns:
            clamped[col] = clamped[col].apply(lambda x: clamp_value(x, mn, mx))
    return clamped


def validate_dataframe(df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """Validate DataFrame against basic schema and ranges.

    - Ensures required columns exist
    - Drops or raises on nulls in required columns
    - Clamps to ranges and removes rows that remain out-of-bounds
    """
    required_cols = [
        "timestamp", "field_id", "soil_ph", "nitrogen", "phosphorus",
        "potassium", "moisture", "temperature", "scenario", "q_value",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with nulls in critical columns
    critical = [c for c in required_cols if c != "scenario"]
    if df[critical].isnull().any().any():
        if drop_invalid:
            df = df.dropna(subset=critical)
        else:
            raise ValueError("Nulls found in critical columns")

    # Clamp and enforce ranges
    df = clamp_dataframe(df)
    for col, (mn, mx) in RANGES.items():
        if col in df.columns:
            mask = (df[col] >= mn) & (df[col] <= mx)
            if drop_invalid:
                df = df[mask]
            else:
                if not mask.all():
                    raise ValueError(f"Out-of-bounds values detected in {col}")
    return df


def ks_compare(real: pd.Series, synthetic: pd.Series) -> float:
    """Return KS-statistic p-value comparing real vs synthetic distribution."""
    real_clean = pd.to_numeric(real, errors="coerce").dropna()
    synth_clean = pd.to_numeric(synthetic, errors="coerce").dropna()
    if len(real_clean) == 0 or len(synth_clean) == 0:
        return 0.0
    _, pvalue = ks_2samp(real_clean, synth_clean)
    return float(pvalue)


