"""
explainability.py
Lightweight explainability utilities. Tries BigQuery ML.EXPLAIN_PREDICT if
available; otherwise computes simple feature importance via leave-one-out
on a provided DataFrame and target.
"""
from __future__ import annotations

from typing import List, Dict
import pandas as pd
import numpy as np
import logging

try:
    from google.cloud import bigquery
    BQ_AVAILABLE = True
except Exception:
    BQ_AVAILABLE = False


def explain_via_bq(model_path: str, table_ref: str) -> pd.DataFrame:
    """Attempt BigQuery ML.EXPLAIN_PREDICT on the given model and table.

    Returns a DataFrame of explanations or raises if unavailable.
    """
    if not BQ_AVAILABLE:
        raise RuntimeError("BigQuery client not available")
    client = bigquery.Client()
    sql = f"""
    SELECT * FROM ML.EXPLAIN_PREDICT(MODEL `{model_path}`, TABLE {table_ref})
    """
    job = client.query(sql)
    return job.result().to_dataframe()


def explain_via_leave_one_out(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Compute simple feature importance by measuring performance drop when
    leaving one feature out using a baseline linear fit.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X = features.select_dtypes(include=[np.number]).fillna(0.0)
    y = pd.to_numeric(target, errors="coerce").fillna(0.0)

    if X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    model = LinearRegression().fit(X, y)
    baseline = r2_score(y, model.predict(X))
    importances: List[Dict[str, float]] = []

    for col in X.columns:
        X_drop = X.drop(columns=[col])
        model_drop = LinearRegression().fit(X_drop, y)
        score_drop = r2_score(y, model_drop.predict(X_drop))
        importances.append({
            "feature": col,
            "importance": max(0.0, baseline - score_drop)
        })

    df_imp = pd.DataFrame(importances).sort_values("importance", ascending=False)
    return df_imp


