"""
utils.py
Generic helpers for file discovery, deterministic scoring, and logging setup.
"""
from __future__ import annotations

from typing import Optional
from pathlib import Path
import hashlib
import numpy as np


def find_first_file(root: str | Path, extensions: tuple[str, ...]) -> Optional[Path]:
    """Return the first file under root with any of the given extensions.

    Search is non-recursive and returns None if not found.
    """
    p = Path(root)
    if not p.exists():
        return None
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in extensions:
            return f
    return None


def deterministic_score(*parts: str, seed: int = 42) -> float:
    """Return a deterministic float in [0,1] derived from provided strings.

    Combines parts and a seed into a stable hash, then maps to [0,1].
    """
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    for part in parts:
        h.update(str(part).encode("utf-8"))
    # Use first 8 bytes as integer
    val = int.from_bytes(h.digest()[:8], byteorder="big")
    return (val % 10_000_000) / 10_000_000.0


def make_field_id(lat: float, lon: float, precision: int = 3) -> str:
    """Create a pseudo field_id from lat/lon with given precision."""
    return f"grid_{round(float(lat), precision)}_{round(float(lon), precision)}"


