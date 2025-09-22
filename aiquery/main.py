"""
main.py
Orchestrate demo: generate synthetic quantum data, ingest to BQ (or local CSV), create embeddings & index, run demo vector search and a forecast.
"""
import os
import argparse
import logging
import importlib
from src.quantum_simulator import generate_quantum_synthetic_data
from src.bigquery_ai_integration import ingest_synthetic_to_bq, create_embeddings_and_index, vector_search, run_forecast
from config.config import DATASET, QUANTUM_SENSORS_TABLE, validate_config

def _print_versions():
    libs = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("qiskit", "qiskit"),
        ("qiskit_aer", "qiskit_aer"),
        ("google-cloud-bigquery", "google.cloud.bigquery"),
    ]
    for name, module in libs:
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "unknown")
            logging.info("%s %s", name, ver)
        except Exception:
            logging.info("%s not installed", name)


def run_pipeline(scenario: str = "soil_chemistry", num_samples: int = 200, field_ids: int = 3, no_quantum: bool = False, offline: bool = False):
    """
    Run the complete quantum-enhanced precision agriculture pipeline.
    
    Args:
        scenario: Agricultural scenario to simulate
        num_samples: Number of synthetic samples to generate
        field_ids: Number of fields to simulate
        no_quantum: If True, disable quantum enhancement (for ablation study)
    """
    logging.info("Starting Quantum-Enhanced Precision Agriculture Pipeline")
    logging.info("Scenario: %s", scenario)
    logging.info("Samples: %s", num_samples)
    logging.info("Fields: %s", field_ids)
    logging.info("Quantum enhancement: %s", 'Disabled' if no_quantum else 'Enabled')
    
    # 1. Generate synthetic quantum sensor data
    logging.info("Step 1: Generating quantum synthetic data...")
    df = generate_quantum_synthetic_data(scenario=scenario, num_samples=num_samples, field_ids=field_ids)
    
    if no_quantum:
        # For ablation study, set all q_values to 0.5 (neutral)
        df['q_value'] = 0.5
        logging.warning("Quantum enhancement disabled for ablation study")
    
    logging.info("Generated %s samples", len(df))
    
    # 2. Ingest to BigQuery or local file
    logging.info("Step 2: Ingesting data...")
    if offline:
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/sample_quantum_synthetic.csv", index=False)
        logging.info("Offline mode: wrote outputs/sample_quantum_synthetic.csv")
        bq_available = False
    else:
        try:
            table_id = ingest_synthetic_to_bq(df, DATASET, QUANTUM_SENSORS_TABLE)
            logging.info("Ingested to BigQuery: %s", table_id)
            bq_available = True
        except Exception as e:
            logging.warning("BigQuery ingest failed: %s", e)
            logging.info("Falling back to local CSV...")
            os.makedirs("outputs", exist_ok=True)
            df.to_csv("outputs/sample_quantum_synthetic.csv", index=False)
            logging.info("Wrote outputs/sample_quantum_synthetic.csv")
            bq_available = False

    # 3. Create embeddings & vector index
    if bq_available and not offline:
        logging.info("Step 3: Creating embeddings and vector index...")
        try:
            embed_table = create_embeddings_and_index(DATASET, QUANTUM_SENSORS_TABLE)
            logging.info("Created embeddings table: %s", embed_table)
        except Exception as e:
            logging.warning("Embedding/index creation failed: %s", e)

        # 4. Demo vector search
        logging.info("Step 4: Performing vector search...")
        try:
            results = vector_search(DATASET, f"{QUANTUM_SENSORS_TABLE}_embeddings", "drought stress in corn fields", top_k=3)
            logging.info("Vector search results:")
            for i, (_, row) in enumerate(results.head(3).iterrows(), 1):
                logging.info("%s. Field %s: pH=%.2f, Score=%s", i, row['field_id'], row['soil_ph'], f"{row.get('score', 'N/A'):.3f}" if 'score' in row else 'N/A')
        except Exception as e:
            logging.warning("Vector search failed: %s", e)

        # 5. Demo forecast
        logging.info("Step 5: Generating yield forecast...")
        try:
            forecast = run_forecast(DATASET, "yield_history")
            logging.info("Forecast results:")
            logging.info("Next 14 days avg: %.1f bushels/acre", forecast['forecast_value'].mean())
            logging.info("Confidence: %.1f - %.1f", forecast['confidence_interval_lower'].mean(), forecast['confidence_interval_upper'].mean())
        except Exception as e:
            logging.warning("Forecast failed: %s", e)
    
    # 6. Summary
    logging.info("Pipeline completed successfully.")
    logging.info("Generated %s quantum synthetic samples", len(df))
    logging.info("Scenario: %s", scenario)
    logging.info("Quantum enhancement: %s", 'Disabled' if no_quantum else 'Enabled')
    if not bq_available:
        logging.info("Data saved to outputs/sample_quantum_synthetic.csv")

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Quantum-Enhanced Precision Agriculture Pipeline")
    parser.add_argument("--scenario", default="soil_chemistry", help="Agricultural scenario to simulate")
    parser.add_argument("--samples", type=int, default=200, help="Number of synthetic samples to generate")
    parser.add_argument("--fields", type=int, default=3, help="Number of fields to simulate")
    parser.add_argument("--no-quantum", action="store_true", help="Disable quantum enhancement (for ablation study)")
    parser.add_argument("--offline", action="store_true", help="Run without BigQuery steps (force local CSV)")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    
    args = parser.parse_args()
    
    # Configure logging and print versions
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    _print_versions()

    # Validate configuration (allow offline run even if placeholders remain)
    config_ok = validate_config()
    if not config_ok and not args.offline:
        logging.warning("Please update config/config.py before running the pipeline")
        return
    
    # Run pipeline
    run_pipeline(
        scenario=args.scenario,
        num_samples=args.samples,
        field_ids=args.fields,
        no_quantum=args.no_quantum,
        offline=args.offline
    )

if __name__ == "__main__":
    main()
