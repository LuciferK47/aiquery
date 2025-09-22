"""
config/config.py
Configuration file with all placeholders that judges need to replace.
"""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# =============================================================================
# JUDGES: REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL VALUES
# =============================================================================

# GCP Project Configuration
PROJECT = "your-project-id"  # Replace with your GCP project ID
DATASET = "aiquery_demo"     # Replace with your preferred dataset name
LOCATION = "US"              # Replace with your preferred BigQuery location

# BigQuery ML Model Names (REPLACE THESE)
EMBEDDING_MODEL = "your_project.your_dataset.embedding_model"      # e.g., "bigqueryml/generative/embedding"
TEXT_MODEL = "your_project.your_dataset.gpt_text_model"           # e.g., "bigqueryml/generative/gpt"
YIELD_FORECAST_MODEL = "your_project.your_dataset.yield_forecast_model"  # e.g., "your_project.your_dataset.yield_model"

# Model Configuration
EMBED_DIM = 1536
MAX_TOKENS = 1024
TEMPERATURE = 0.7

# Quantum Simulation Configuration
QUANTUM_SEED = 42
QUANTUM_SHOTS = 64
QUANTUM_QUBITS = 3

# Data Configuration
DEFAULT_SAMPLES = 200
DEFAULT_FIELDS = 3
CACHE_SIZE = 1024

# Table Names
QUANTUM_SENSORS_TABLE = "quantum_sensors"
EMBEDDINGS_TABLE = "quantum_sensors_embeddings"
VECTOR_INDEX_NAME = "quantum_sensors_vec_idx"

# =============================================================================
# ENVIRONMENT VARIABLES (automatically loaded)
# =============================================================================

# Load .env from project root if available
if load_dotenv is not None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))

# Override with environment variables if available
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", PROJECT)
DATASET = os.environ.get("BQ_DATASET", DATASET)
LOCATION = os.environ.get("BIGQUERY_LOCATION", LOCATION)

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate that all required placeholders have been replaced."""
    import logging

    placeholders = [
        ("PROJECT", PROJECT),
        ("EMBEDDING_MODEL", EMBEDDING_MODEL),
        ("TEXT_MODEL", TEXT_MODEL),
        ("YIELD_FORECAST_MODEL", YIELD_FORECAST_MODEL)
    ]

    missing_replacements = []
    for name, value in placeholders:
        if isinstance(value, str) and ("your_project" in value or "your_dataset" in value):
            missing_replacements.append(f"{name} = {value}")

    if missing_replacements:
        logging.warning("WARNING: Please replace these placeholders in config/config.py:")
        for item in missing_replacements:
            logging.warning("   %s", item)
        logging.warning("The demo will work with CSV fallback, but BigQuery features will be limited.")
        return False

    logging.info("Configuration validated - all placeholders replaced!")
    return True

if __name__ == "__main__":
    validate_config()

# -----------------------------------------------------------------------------
# Additional application configuration used by modules in src/
# -----------------------------------------------------------------------------
from types import SimpleNamespace

# Path to a simple survey text used by QuantumDataProcessor demo
DEFAULT_SURVEY_FILE = str(Path(__file__).resolve().parents[1] / "data" / "survey.txt")

# Expose an object named `config` expected by src/data_processor.py
config = SimpleNamespace(
    survey_file=os.environ.get("SURVEY_FILE", DEFAULT_SURVEY_FILE)
)

# Visualization config expected by src/visualization_engine.py
viz_config = SimpleNamespace(
    output_dir=str(Path(__file__).resolve().parents[1] / "outputs"),
    color_palette=["#2E86AB", "#F6C85F", "#6FB07F", "#D7263D", "#9D4EDD"],
    map_center=[20.0, 0.0],
    map_zoom=2
)

# Minimal quantum sensor applications catalog used by both data processing and visualization
QUANTUM_SENSOR_APPLICATIONS = {
    "gravitational_wave_detection": {
        "description": "Quantum sensors for gravitational wave detection and astrophysical observation",
        "domains": ["research", "astrophysics", "national_labs"],
        "market_potential": "medium",
    },
    "magnetic_field_sensing": {
        "description": "NV-center magnetometry and atomic magnetometers for biomedical and geophysical sensing",
        "domains": ["biomedical", "geoscience", "defense"],
        "market_potential": "high",
    },
    "quantum_clock_synchronization": {
        "description": "Optical lattice clocks and quantum timekeeping for navigation and finance",
        "domains": ["telecom", "finance", "navigation"],
        "market_potential": "high",
    },
    "quantum_imaging": {
        "description": "Quantum-enhanced imaging and sensing for low-light and high-resolution applications",
        "domains": ["medical", "manufacturing", "security"],
        "market_potential": "medium",
    },
    "environmental_monitoring": {
        "description": "Quantum sensors for environmental monitoring and precision agriculture",
        "domains": ["agriculture", "environment"],
        "market_potential": "high",
    },
}