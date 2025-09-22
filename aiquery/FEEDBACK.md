BigQuery AI Feedback

Vector search and embeddings
- Ann tier and feature availability for VECTOR INDEX vary by region and tier. A clearer compatibility matrix in docs and surfaced in errors would reduce trial‑and‑error.
- ML.DOT_PRODUCT fallback is practical, but performance is not comparable to ANN. A managed fallback with approximate search options exposed in SQL would help.

Model management
- Versioning in SQL is powerful, but discoverability in the UI could improve. A registry‑style view with lineage and promotion status would streamline audits.
- Stronger guardrails for schema drift between training and prediction would be useful, with warnings at query compile time.

Operational concerns
- More explicit quotas and rate limits for ML.GENERATE_EMBEDDING and ML.GENERATE_TEXT would help plan batch jobs.
- Clearer guidance on embedding dimensionality and tokenization across supported models would reduce mismatches when building vector indexes.

Developer experience
- Better local emulation guidance for AI.FORECAST and vector index features would smooth offline development.
- Example repos that pair SQL with Python orchestration (Airflow, Cloud Workflows) would shorten onboarding time.


