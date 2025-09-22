"""
stress_test.py
Simulate extreme conditions and compare system outputs to baseline.
"""
from __future__ import annotations

import logging
import pandas as pd

from src.quantum_simulator import generate_quantum_synthetic_data
from src.calibration import clamp_dataframe


def run_stress_test(samples: int = 200) -> pd.DataFrame:
    """Run a simple stress test comparing baseline vs drought scenario."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base = generate_quantum_synthetic_data(scenario="baseline", num_samples=samples, field_ids=3)
    drought = generate_quantum_synthetic_data(scenario="extreme_drought", num_samples=samples, field_ids=3)

    # Impose stress by lowering moisture and increasing temperature
    drought = drought.copy()
    drought["moisture"] = (drought["moisture"] * 0.5).clip(lower=0.0)
    drought["temperature"] = drought["temperature"] + 4.0

    base = clamp_dataframe(base)
    drought = clamp_dataframe(drought)

    # Compare simple aggregates
    summary = pd.DataFrame({
        "metric": ["avg_moisture", "avg_temperature", "avg_ph"],
        "baseline": [base["moisture"].mean(), base["temperature"].mean(), base["soil_ph"].mean()],
        "drought": [drought["moisture"].mean(), drought["temperature"].mean(), drought["soil_ph"].mean()],
    })
    summary["delta"] = summary["drought"] - summary["baseline"]
    logging.info("Stress test summary:\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_stress_test()


