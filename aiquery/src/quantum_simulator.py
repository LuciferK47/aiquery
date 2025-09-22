"""
src/quantum_simulator.py
Generate reproducible synthetic 'quantum sensor' readings using Qiskit.
This is a fast, hybrid simulation used to produce plausible, diverse sensor streams.
"""
from typing import Dict
import numpy as np
import pandas as pd
import datetime
from functools import lru_cache
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import QUANTUM_SEED, QUANTUM_SHOTS, QUANTUM_QUBITS, CACHE_SIZE

# Set deterministic seed
np.random.seed(QUANTUM_SEED)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a numeric value between provided bounds."""
    return max(min_value, min(max_value, float(value)))

@lru_cache(maxsize=CACHE_SIZE)
def _cached_quantum_sample(seed: int, num_qubits: int = QUANTUM_QUBITS, shots: int = QUANTUM_SHOTS) -> float:
    """
    Return deterministic quantum sample for a (seed, num_qubits) tuple.
    Cached to avoid repeated simulator runs for same time windows.
    """
    qc = QuantumCircuit(num_qubits)
    # Add some parameterized structure to vary based on seed/time
    for i in range(num_qubits):
        qc.h(i)
    qc.barrier()
    qc.measure_all()
    
    simulator = AerSimulator()
    compiled = transpile(qc, simulator, optimization_level=0)
    job = simulator.run(compiled, shots=shots, seed_simulator=seed, seed_transpiler=seed)
    result = job.result().get_counts()
    
    # Convert counts distribution to a float in [0,1]
    values = []
    for bitstring, count in result.items():
        val = int(bitstring, 2) / (2**num_qubits - 1)
        values.append(val * (count / shots))
    return float(sum(values))

def _simple_quantum_noise_sample(num_qubits: int = QUANTUM_QUBITS, shots: int = QUANTUM_SHOTS) -> float:
    """
    Build a tiny circuit whose measurement statistics we map to a numeric value.
    This keeps runtime short while producing varied outputs.
    """
    # Use current timestamp as seed for variation, but cache by time window
    current_time = int(datetime.datetime.now().timestamp())
    time_window = current_time // 900  # 15-minute windows
    return _cached_quantum_sample(time_window, num_qubits, shots)

def generate_quantum_synthetic_data(scenario: str = "soil_chemistry", num_samples: int = 500, field_ids: int = 5) -> pd.DataFrame:
    """
    Generate a DataFrame with synthetic sensor readings.
    Columns: timestamp, field_id, soil_ph, nitrogen, phosphorus, potassium, moisture, temperature, scenario, q_value
    The quantum-derived q_value injects structured, complex variability.
    """
    # Use a stable 15-minute aligned end time to ensure reproducibility across calls
    base_end = pd.Timestamp.now(tz='UTC').floor('15min')
    timestamps = pd.date_range(end=base_end, periods=num_samples, freq='15min', tz='UTC')
    records = []
    
    for i, ts in enumerate(timestamps):
        field_id = f"field_{(i % field_ids) + 1}"
        
        # Use deterministic time-based seed for reproducibility (15-minute windows)
        time_seed = int(ts.timestamp()) // 900
        q_val = _cached_quantum_sample(time_seed, QUANTUM_QUBITS, QUANTUM_SHOTS)

        # Local RNG seeded deterministically per record to ensure reproducible draws
        local_seed = QUANTUM_SEED + time_seed * 1000 + i
        rng = np.random.default_rng(local_seed)
        
        # Generate sensor readings with quantum-influenced variability
        base_ph = 6.5 + (q_val - 0.5) * 0.6
        nitrogen = rng.normal(10 + q_val * 2, 1.5)
        phosphorus = rng.normal(5 + q_val * 1.2, 1.0)
        potassium = rng.normal(8 + q_val * 1.4, 1.2)
        moisture = rng.normal(30 + q_val * 5, 5.0)
        temp = rng.normal(20 + (q_val - 0.5) * 6, 2.5)

        # Clamp to realistic agronomic ranges
        base_ph = _clamp(base_ph, 3.5, 9.0)
        nitrogen = _clamp(nitrogen, 0.0, 100.0)
        phosphorus = _clamp(phosphorus, 0.0, 50.0)
        potassium = _clamp(potassium, 0.0, 100.0)
        moisture = _clamp(moisture, 0.0, 100.0)
        temp = float(temp)
        
        records.append({
            # Store timestamp as ISO string to preserve round-trippable equality via CSV
            "timestamp": ts.isoformat(),
            "field_id": field_id,
            "soil_ph": round(float(base_ph), 3),
            "nitrogen": round(float(nitrogen), 3),
            "phosphorus": round(float(phosphorus), 3),
            "potassium": round(float(potassium), 3),
            "moisture": round(float(moisture), 3),
            "temperature": round(float(temp), 3),
            "scenario": scenario,
            "q_value": float(q_val)
        })
    
    df = pd.DataFrame.from_records(records)
    return df

if __name__ == "__main__":
    # quick CLI for local dev
    df = generate_quantum_synthetic_data(num_samples=50, field_ids=3)
    print(df.head())
    df.to_csv("outputs/sample_quantum_synthetic.csv", index=False)
