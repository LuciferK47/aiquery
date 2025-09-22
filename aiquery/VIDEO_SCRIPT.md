Video Demo Script

Duration: 4–6 minutes

1) Hook and Problem (0:00–0:30)
- Farms lose yield because rare, high‑impact soil events are missing from historical data. We simulate those with quantum‑inspired data and run AI end‑to‑end in BigQuery.

2) Architecture Overview (0:30–1:00)
- Flow: Quantum simulation -> BigQuery ingestion -> Embeddings and vector search -> Text recommendations -> Forecast.
- Offline‑first: If GCP is not available, the pipeline runs locally and writes to output/.

3) Local Real‑Data Preprocessing (1:00–1:45)
- Run: python -m aiquery.src.real_data_preprocessor
- Show: output/processed_data.csv and output/model_summary.txt
- Explain schema: timestamp, field_id, soil_ph, nitrogen, phosphorus, potassium, moisture, temperature, q_value with year aggregation for soil moisture.
- Note validation and safe defaults for missing values.

3a) What the outputs represent (add while screen‑sharing the files)
- output/processed_data.csv: Unified, cleaned table joining SoilHealthDB properties with aggregated soil moisture. Each row is a field‑year observation ready for ML or SQL analysis.
- output/model_summary.txt: Coefficient and intercept from a simple linear model trained on real merged data (moisture vs a proxy target like potassium when yield is absent). Shows end‑to‑end trainability from local data.
- Notebook outputs (notebooks/demo_agri_workflow.executed.ipynb): Executed notebook illustrating data inspection, basic charts, and example queries. Use this to show plots of distributions and any time series summaries derived from the same schema.

4) BigQuery AI Usage (1:45–3:15)
- Designed to run fully in BigQuery when credentials exist. Three pillars:
  - Embeddings and vector search: ML.GENERATE_EMBEDDING and similarity search (with ML.DOT_PRODUCT fallback).
  - Text generation: ML.GENERATE_TEXT using soil features.
  - Forecasting: AI.FORECAST over yield history.
- When credentials are not available, persist CSVs locally with identical schema so enabling BigQuery is a simple switch.

5) Innovation and Impact (3:15–4:15)
- Quantum synthetic data populates edge cases; models learn rare events.
- Measured improvements in similarity search and forecasts for our scenarios.
- SQL‑first GenAI and search directly in the warehouse.

6) Quick Judge Runbook (4:15–5:00)
- Offline: pip install -r requirements.txt, then python -m aiquery.src.real_data_preprocessor to generate outputs.
- BigQuery: set the 4 placeholders in config/config.py, then python main.py.

7) Close and Call to Action (5:00–5:30)
- Quantum‑enhanced, SQL‑native AI for agriculture. The repo is public and ready to run.


