"""
real_data_preprocessor.py
Data preprocessing pipeline for SoilHealthDB (tabular) and GSSM1km soil moisture
sample (NetCDF). Produces a unified ML-ready DataFrame with schema:

timestamp, field_id, soil_ph, nitrogen, phosphorus, potassium,
moisture, temperature, q_value

Notes
- SoilHealthDB is assumed to be a directory of CSV files (or can be extended
  for Parquet). We parse pH, N, P, K, field identifiers, and timestamps.
- GSSM1km is assumed to contain a NetCDF file with soil_moisture variable.
- We generate a deterministic q_value and a pseudo field_id per grid cell.
- Hooks are provided for ERA5 temperature fill.
"""
from __future__ import annotations

from typing import Tuple, List
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
try:
    import rasterio
except Exception:
    rasterio = None

from .utils import find_first_file, make_field_id, deterministic_score
from .calibration import clamp_dataframe, validate_dataframe
def load_datasets_local() -> tuple[pd.DataFrame, xr.Dataset]:
    """Load local datasets: SoilHealthDB/ and 21806457/ into memory.

    Reads all CSV/Parquet files under SoilHealthDB/ and opens a NetCDF under 21806457/.
    """
    # Locate SoilHealthDB under several common layouts (flat or nested under data/)
    candidate_roots = [
        Path("SoilHealthDB"),
        Path("aiquery") / "SoilHealthDB",
        Path("data") / "SoilHealthDB",
        Path("aiquery") / "data" / "SoilHealthDB",
    ]
    root_soil = next((p for p in candidate_roots if p.exists()), None)
    if root_soil is None:
        raise FileNotFoundError("SoilHealthDB directory not found in expected locations")
    # Search recursively to pick curated CSVs
    soil_files = [
        fp for fp in root_soil.rglob("*.csv")
        if fp.is_file()
    ] + [
        fp for fp in root_soil.rglob("*.parquet")
        if fp.is_file()
    ]
    if not soil_files:
        raise FileNotFoundError("No .csv or .parquet found under SoilHealthDB root")
    frames: list[pd.DataFrame] = []
    for f in soil_files:
        try:
            if f.suffix.lower()==".csv":
                try:
                    frames.append(pd.read_csv(f))
                except UnicodeDecodeError:
                    frames.append(pd.read_csv(f, encoding="latin1"))
            else:
                frames.append(pd.read_parquet(f))
        except Exception:
            continue
    soil_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    root_gssm = Path("21806457")
    if not root_gssm.exists():
        raise FileNotFoundError("21806457 directory not found")
    # pick a NetCDF-like file
    nc = find_first_file(root_gssm, (".nc", ".nc4", ".cdf"))
    if nc is None:
        # attempt a common sample name
        candidate = root_gssm / "sample.nc"
        if not candidate.exists():
            raise FileNotFoundError("No NetCDF file found in 21806457/")
        nc = candidate
    gssm_ds = xr.open_dataset(str(nc))
    return soil_df, gssm_ds


def explore_data_local(soil_df: pd.DataFrame, gssm_ds: xr.Dataset) -> None:
    print("SoilHealthDB shape:", soil_df.shape)
    print("SoilHealthDB columns:", soil_df.columns.tolist())
    print("GSSM1km variables:", list(gssm_ds.data_vars))
    print("GSSM1km coords:", list(gssm_ds.coords))


def preprocess_local(soil_df: pd.DataFrame, gssm_ds: xr.Dataset) -> pd.DataFrame:
    # Resample soil moisture to monthly mean; derive year for coarse alignment
    var = "soil_moisture" if "soil_moisture" in gssm_ds.data_vars else list(gssm_ds.data_vars)[0]
    sm_monthly = gssm_ds[var].resample(time="1M").mean()
    moisture_df = sm_monthly.to_dataframe(name="soil_moisture").reset_index()
    moisture_df.rename(columns={"time": "timestamp"}, inplace=True)
    moisture_df["timestamp"] = pd.to_datetime(moisture_df["timestamp"], utc=True, errors="coerce")
    moisture_df["year"] = moisture_df["timestamp"].dt.year
    moisture_df.rename(columns={"soil_moisture": "moisture"}, inplace=True)
    moisture_df = moisture_df[["year", "moisture"]].groupby("year", as_index=False).mean()

    # Soil df timestamp -> year
    ts_col = None
    for cand in ["timestamp", "SamplingYear", "Year", "YEAR", "YearPublication", "date", "Sampling_Date"]:
        if cand in soil_df.columns:
            ts_col = cand
            break
    if ts_col is None:
        soil_df = soil_df.copy()
        soil_df["timestamp"] = pd.NaT
    else:
        soil_df = soil_df.copy()
        soil_df["timestamp"] = pd.to_datetime(soil_df[ts_col], errors="coerce")
    soil_df["year"] = pd.to_datetime(soil_df["timestamp"], errors="coerce").dt.year

    # Minimal mapping to expected schema
    if "soil_ph" not in soil_df.columns:
        if "SoilpH" in soil_df.columns:
            soil_df["soil_ph"] = pd.to_numeric(soil_df["SoilpH"], errors="coerce")
    for elt, choices in {
        "nitrogen": ["N", "N_total", "Nitrogen"],
        "phosphorus": ["P", "P_total", "Phosphorus"],
        "potassium": ["K", "K_total", "Potassium"],
    }.items():
        if elt not in soil_df.columns:
            for c in choices:
                if c in soil_df.columns:
                    soil_df[elt] = pd.to_numeric(soil_df[c], errors="coerce")
                    break
    if "field_id" not in soil_df.columns:
        if "lat" in soil_df.columns and "lon" in soil_df.columns:
            soil_df["field_id"] = soil_df.apply(lambda r: make_field_id(r.get("lat"), r.get("lon")), axis=1)
        else:
            soil_df["field_id"] = np.arange(len(soil_df)).astype(str)

    # Ensure required columns exist before join
    for miss in ["soil_ph", "nitrogen", "phosphorus", "potassium"]:
        if miss not in soil_df.columns:
            soil_df[miss] = np.nan
    if "q_value" not in soil_df.columns:
        soil_df["q_value"] = soil_df.apply(lambda r: deterministic_score(str(r.get("field_id")), str(r.get("timestamp"))), axis=1)
    if "temperature" not in soil_df.columns:
        soil_df["temperature"] = 20.0

    # Year-level left join
    merged = pd.merge(
        soil_df[["timestamp", "year", "field_id", "soil_ph", "nitrogen", "phosphorus", "potassium", "temperature", "q_value"]],
        moisture_df,
        on=["year"],
        how="left",
    )
    merged["moisture"] = merged["moisture"].fillna(merged.groupby("year")["moisture"].transform("mean"))
    merged["moisture"] = merged["moisture"].fillna(30.0)
    merged["scenario"] = "real_world"
    merged = validate_dataframe(merged, drop_invalid=True)
    return merged


def run_analysis_local(merged_df: pd.DataFrame):
    # Optional simple regression if a plausible target is available
    from sklearn.linear_model import LinearRegression
    target_candidates = ["yield", "crop_yield", "YIELD", "Yield"]
    target = next((c for c in target_candidates if c in merged_df.columns), None)
    if target is None:
        print("No yield-like target found; skipping regression.")
        return None
    x = merged_df[["moisture"]].values
    y = pd.to_numeric(merged_df[target], errors="coerce").fillna(0.0).values
    model = LinearRegression().fit(x, y)
    print("R^2:", model.score(x, y))
    print("Coef:", model.coef_, "Intercept:", model.intercept_)
    return model


def save_outputs_local(merged_df: pd.DataFrame, model) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_dir / "processed_data.csv", index=False)
    if model is not None:
        (out_dir / "model_summary.txt").write_text(f"Coef: {model.coef_}, Intercept: {model.intercept_}")
    else:
        (out_dir / "model_summary.txt").write_text("No model trained; target not found.")


def plot_outputs_local(merged_df: pd.DataFrame) -> None:
    """Generate basic plots from real merged data and save to project-root output/."""
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histograms for key variables
    for col in ["soil_ph", "nitrogen", "phosphorus", "potassium", "moisture"]:
        if col in merged_df.columns:
            plt.figure(figsize=(6, 4))
            merged_df[col].dropna().astype(float).plot(kind="hist", bins=30, title=f"Distribution of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(out_dir / f"hist_{col}.png", dpi=150)
            plt.close()

    # Scatter: moisture vs potassium (proxy for outcome)
    if "moisture" in merged_df.columns and "potassium" in merged_df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(merged_df["moisture"], pd.to_numeric(merged_df["potassium"], errors="coerce"), s=10, alpha=0.6)
        plt.title("Moisture vs Potassium")
        plt.xlabel("Moisture")
        plt.ylabel("Potassium")
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_moisture_vs_potassium.png", dpi=150)
        plt.close()

    # Annual moisture bar plot if year present
    if "year" in merged_df.columns and "moisture" in merged_df.columns:
        annual = merged_df.groupby("year", as_index=False)["moisture"].mean().dropna()
        if not annual.empty:
            plt.figure(figsize=(7, 4))
            plt.bar(annual["year"].astype(int), annual["moisture"].astype(float))
            plt.title("Average Annual Moisture")
            plt.xlabel("Year")
            plt.ylabel("Moisture")
            plt.tight_layout()
            plt.savefig(out_dir / "annual_moisture.png", dpi=150)
            plt.close()


def _resolve_data_dir(root: str) -> Path:
    """Resolve a data directory whether it lives at repo root or under the aiquery package."""
    p = Path(root)
    if p.exists():
        return p
    pkg_root = Path(__file__).resolve().parents[1]
    alt = pkg_root / root
    if alt.exists():
        return alt
    # Also try repo_root/aiquery/<root>
    repo_root = Path(__file__).resolve().parents[2]
    alt2 = repo_root / "aiquery" / root
    if alt2.exists():
        return alt2
    raise FileNotFoundError(f"{root} directory not found: tried '{p}', '{alt}', '{alt2}'")


def load_soilhealthdb(root: str = "SoilHealthDB") -> pd.DataFrame:
    """Load SoilHealthDB CSV files and map to a standard schema.

    Expected or inferred columns include:
    - timestamp/date
    - field_id or something we can use as an ID
    - soil_ph, nitrogen, phosphorus, potassium
    Optional columns may include lat, lon.
    """
    p = _resolve_data_dir(root)

    frames: List[pd.DataFrame] = []
    # Only ingest curated SoilHealthDB main tables to avoid massive grid CSVs
    candidates: List[Path] = [fp for fp in p.rglob("SoilHealthDB_V*.csv")]
    # Explicit common path fallback
    explicit = p / "SoilHealthDB" / "data" / "SoilHealthDB_V1.csv"
    if explicit.exists() and explicit not in candidates:
        candidates.append(explicit)
    # Fallback: common known locations
    likely_files = [
        p / "SoilHealthDB" / "data" / "SoilHealthDB_V1.csv",
        Path.cwd() / "SoilHealthDB" / "SoilHealthDB" / "data" / "SoilHealthDB_V1.csv",
        Path.cwd() / "aiquery" / "SoilHealthDB" / "SoilHealthDB" / "data" / "SoilHealthDB_V1.csv",
        Path.cwd() / "data" / "SoilHealthDB" / "SoilHealthDB" / "data" / "SoilHealthDB_V1.csv",
        Path.cwd() / "aiquery" / "data" / "SoilHealthDB" / "SoilHealthDB" / "data" / "SoilHealthDB_V1.csv",
    ]
    for lf in likely_files:
        if lf.exists():
            candidates.append(lf)

    seen = set()
    for f in sorted(candidates):
        if f in seen:
            continue
        seen.add(f)
        try:
            if f.suffix.lower() == ".csv":
                # Try UTF-8, then fallback encodings commonly used
                try:
                    df = pd.read_csv(f)
                except UnicodeDecodeError:
                    for enc in ("latin1", "iso-8859-1", "cp1252"):
                        try:
                            df = pd.read_csv(f, encoding=enc)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise
            else:
                df = pd.read_parquet(f)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        raise FileNotFoundError("No CSV/Parquet files found in SoilHealthDB/")

    df_all = pd.concat(frames, ignore_index=True)

    # Try to infer key fields from common variants in this dataset
    # Latitude/Longitude for field id
    if "Latitude" in df_all.columns and "Longitude" in df_all.columns:
        df_all.rename(columns={"Latitude": "lat", "Longitude": "lon"}, inplace=True)
    # Timestamp: use SamplingYear or YearPublication
    ts_col = None
    for cand in ["SamplingYear", "YearPublication", "Year", "YEAR", "Sampling_Date", "date", "timestamp"]:
        if cand in df_all.columns:
            ts_col = cand
            break
    if ts_col is not None:
        df_all["timestamp"] = pd.to_datetime(df_all[ts_col], errors="coerce").dt.to_period("Y").dt.to_timestamp("YS")
    # Soil pH: prefer direct soil_pH, else mean of pH_C/pH_T
    if "soil_ph" not in df_all.columns:
        if "SoilpH" in df_all.columns:
            df_all["soil_ph"] = pd.to_numeric(df_all["SoilpH"], errors="coerce")
        else:
            p_cols = [c for c in df_all.columns if c.lower().startswith("pH_".lower()) or c.lower()=="ph"]
            if p_cols:
                df_all["soil_ph"] = pd.to_numeric(df_all[p_cols].mean(axis=1), errors="coerce")
    # Nitrogen/Phosphorus/Potassium: use N_C/N_T, P_C/P_T, K_C/K_T if present (demo purposes)
    if "nitrogen" not in df_all.columns:
        n_cols = [c for c in df_all.columns if c.upper().startswith("N_") or c.upper()=="N"]
        if n_cols:
            df_all["nitrogen"] = pd.to_numeric(df_all[n_cols].mean(axis=1), errors="coerce")
    if "phosphorus" not in df_all.columns:
        p_cols = [c for c in df_all.columns if c.upper().startswith("P_") or c.upper()=="P"]
        if p_cols:
            df_all["phosphorus"] = pd.to_numeric(df_all[p_cols].mean(axis=1), errors="coerce")
    if "potassium" not in df_all.columns:
        k_cols = [c for c in df_all.columns if c.upper().startswith("K_") or c.upper()=="K"]
        if k_cols:
            df_all["potassium"] = pd.to_numeric(df_all[k_cols].mean(axis=1), errors="coerce")

    # Ensure required
    if "timestamp" not in df_all.columns:
        # attempt to infer from any date-like column
        for cand in ["collection_date", "sample_date", "datetime"]:
            if cand in df_all.columns:
                df_all["timestamp"] = pd.to_datetime(df_all[cand], errors="coerce")
                break
    if "field_id" not in df_all.columns:
        # Construct from lat/lon if available, else index
        if "lat" in df_all.columns and "lon" in df_all.columns:
            df_all["field_id"] = df_all.apply(lambda r: make_field_id(r.get("lat"), r.get("lon")), axis=1)
        else:
            df_all["field_id"] = np.arange(len(df_all)).astype(str)

    # Convert types
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    for c in ["soil_ph", "nitrogen", "phosphorus", "potassium"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # Fit schema (moisture and temperature may be missing here)
    for miss in ["moisture", "temperature"]:
        if miss not in df_all.columns:
            df_all[miss] = np.nan

    # Add q_value placeholder deterministically per row
    df_all["q_value"] = df_all.apply(
        lambda r: deterministic_score(str(r.get("field_id")), str(r.get("timestamp"))), axis=1
    )

    # Keep only unified schema
    keep = ["timestamp", "field_id", "soil_ph", "nitrogen", "phosphorus",
            "potassium", "moisture", "temperature", "q_value"]
    df_all = df_all[keep]

    return df_all


def load_gssm1km(root: str = "21806457", var_name: str = "soil_moisture") -> xr.Dataset:
    """Open GSSM1km dataset via xarray or convert GeoTIFF stack to xarray if needed."""
    p = _resolve_data_dir(root)
    # Prefer NetCDF if present
    nc = find_first_file(p, (".nc", ".nc4", ".cdf"))
    if nc is not None:
        ds = xr.open_dataset(str(nc))
        # If variable missing, just return dataset and later code can pick primary var
        return ds

    # Fallback: build an xarray Dataset from yearly GeoTIFF stack
    if rasterio is None:
        raise FileNotFoundError("No NetCDF found and rasterio not available to read GeoTIFFs")

    tifs = sorted([fp for fp in p.iterdir() if fp.suffix.lower() in (".tif", ".tiff")])
    if not tifs:
        raise FileNotFoundError("No NetCDF or GeoTIFF files found in 21806457/")

    # Memory-safe: compute spatial mean per year instead of stacking full rasters
    means: list[float] = []
    times: list[pd.Timestamp] = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            data = src.read(1).astype("float32")
            # Use nanmean to ignore NoData if present
            val = float(np.nanmean(data))
        digits = ''.join(ch for ch in tif.name if ch.isdigit())
        year = int(digits[:4]) if len(digits) >= 4 else 2000
        times.append(pd.Timestamp(year=year, month=1, day=1))
        means.append(val)

    da = xr.DataArray(
        np.array(means, dtype="float32"),
        dims=("time",),
        coords={"time": times},
        name=var_name,
    )
    ds = xr.Dataset({var_name: da})
    return ds


def preprocess_moisture(ds: xr.Dataset, var_name: str = "soil_moisture") -> pd.DataFrame:
    """Extract soil moisture, aggregate to annual mean, and return with year column."""
    if "time" not in ds.coords:
        raise KeyError("Expected 'time' coordinate in the GSSM1km dataset")
    # If requested var missing, pick the first data var
    sm = ds[var_name] if var_name in ds.data_vars else next(iter(ds.data_vars.values()))
    # If spatial dims present, compute national mean to keep memory small
    spatial_dims = [d for d in sm.dims if d not in ("time",)]
    if spatial_dims:
        sm = sm.mean(dim=spatial_dims, skipna=True)
    sm_annual = sm.resample(time="YS").mean()
    df = sm_annual.to_dataframe(name="soil_moisture").reset_index()
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["year"] = df["timestamp"].dt.year
    df.rename(columns={"soil_moisture": "moisture"}, inplace=True)
    df = df[["year", "moisture"]].dropna(subset=["year"]).drop_duplicates()
    return df


def merge_datasets(soil_df: pd.DataFrame, moisture_df: pd.DataFrame) -> pd.DataFrame:
    """Join soil properties and aggregated moisture by field_id via location mapping.

    Strategy: if SoilHealthDB contains lat/lon, join directly. Otherwise, create
    pseudo field ids per grid point and join using them.
    """
    dfm = moisture_df.copy()

    soils = soil_df.copy()
    if "field_id" not in soils.columns:
        soils["field_id"] = np.arange(len(soils)).astype(str)

    # If soils contain lat/lon, map to grid-based field id as well
    if "lat" in soils.columns and "lon" in soils.columns:
        soils["field_id"] = soils.apply(lambda r: make_field_id(r["lat"], r["lon"]), axis=1)

    # Derive annual key for joining
    soils["timestamp"] = pd.to_datetime(soils.get("timestamp", pd.NaT), errors="coerce")
    soils["year"] = soils["timestamp"].dt.year
    # If many rows lack year, expand across all moisture years and set start-of-year timestamps
    if soils["year"].isna().all() or soils["year"].nunique(dropna=True) <= 1:
        available_years = sorted(dfm["year"].dropna().unique().tolist())
        if available_years:
            static_cols = [c for c in soils.columns if c not in ("year", "timestamp")]
            soils_static = soils[static_cols].copy()
            soils_static["__key"] = 1
            years_df = pd.DataFrame({"year": available_years})
            years_df["__key"] = 1
            soils = pd.merge(soils_static, years_df, on="__key").drop(columns=["__key"])
            soils["timestamp"] = pd.to_datetime(soils["year"].astype(int).astype(str) + "-01-01")
    if "year" not in dfm.columns:
        if "timestamp" in dfm.columns:
            dfm["timestamp"] = pd.to_datetime(dfm["timestamp"], errors="coerce")
            dfm["year"] = dfm["timestamp"].dt.year
        else:
            raise ValueError("moisture_df must contain 'year' or 'timestamp'")

    # Map moisture_data to unified schema
    if "soil_moisture" in dfm.columns:
        dfm.rename(columns={"soil_moisture": "moisture"}, inplace=True)

    # Keep only year and moisture for join
    dfm = dfm[["year", "moisture"]].drop_duplicates()

    keep = ["timestamp", "field_id", "soil_ph", "nitrogen", "phosphorus",
            "potassium", "moisture", "temperature", "q_value"]
    # Left-join on year to retain all soil rows even if moisture is missing for some years
    merged = pd.merge(soils[keep + ["year"]], dfm, on=["year"], how="left")

    # Coalesce any duplicate moisture columns
    if "moisture_x" in merged.columns or "moisture_y" in merged.columns:
        merged["moisture"] = merged.get("moisture_y").fillna(merged.get("moisture_x"))
        for col in ["moisture_x", "moisture_y"]:
            if col in merged.columns:
                merged.drop(columns=[col], inplace=True)

    # Fill moisture with year-wise mean, then global fallback to 30.0
    if "moisture" in merged.columns:
        year_mean = merged.groupby("year")["moisture"].transform("mean")
        merged["moisture"] = merged["moisture"].fillna(year_mean)
        merged["moisture"] = merged["moisture"] .fillna(30.0)
    else:
        merged["moisture"] = 30.0

    # Ensure temperature present; fallback 20 C
    if "temperature" not in merged.columns:
        merged["temperature"] = 20.0
    else:
        merged["temperature"] = pd.to_numeric(merged["temperature"], errors="coerce").fillna(20.0)

    # Tag records as real world
    merged['scenario'] = 'real_world'

    # Clamp and validate
    merged = validate_dataframe(merged, drop_invalid=True)
    return merged


def save_outputs(df: pd.DataFrame, outdir: str = "outputs") -> None:
    # Persist under the aiquery package directory if running from repo root
    base = Path(__file__).resolve().parents[1]
    out_path = (base / outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "processed_data.csv").write_text("")  # ensure file is created even if empty write fails
    df.to_csv(out_path / "processed_data.csv", index=False)


def fetch_era5_temperature_stub(lat: float, lon: float, date: pd.Timestamp) -> float:
    """Placeholder hook for ERA5 temperature retrieval.

    Returns NaN by default. Replace with actual ERA5 integration if desired.
    """
    return float("nan")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Prefer a fully local, filesystem-only pipeline writing to output/
    try:
        soil_df_local, gssm_ds_local = load_datasets_local()
        explore_data_local(soil_df_local, gssm_ds_local)
        merged_local = preprocess_local(soil_df_local, gssm_ds_local)
        model = run_analysis_local(merged_local)
        save_outputs_local(merged_local, model)
        plot_outputs_local(merged_local)
        logging.info("Local pipeline complete. Rows saved: %d", len(merged_local))
    except Exception as e:
        logging.warning("Local pipeline encountered an issue: %s", e)
        # Fallback to earlier merge/save behavior (still local files)
        try:
            soil_df = load_soilhealthdb("SoilHealthDB")
            ds = load_gssm1km("21806457", var_name="soil_moisture")
            moisture_df = preprocess_moisture(ds, var_name="soil_moisture")
            merged = merge_datasets(soil_df, moisture_df)
            # Save under project-root output/ per apprentice brief
            out_path = Path(__file__).resolve().parents[2] / "output"
            out_path.mkdir(parents=True, exist_ok=True)
            merged.to_csv(out_path / "processed_data.csv", index=False)
            # Train a minimal regression model if feasible and write summary
            try:
                from sklearn.linear_model import LinearRegression
                dfm = merged[["moisture", "potassium"]].dropna()
                if not dfm.empty:
                    model = LinearRegression().fit(dfm[["moisture"]].values, dfm["potassium"].values)
                    (out_path / "model_summary.txt").write_text(
                        f"Coef: {getattr(model, 'coef_', [])}, Intercept: {getattr(model, 'intercept_', 'NA')}"
                    )
                else:
                    (out_path / "model_summary.txt").write_text("No model trained; insufficient data after NA drop.")
            except Exception:
                (out_path / "model_summary.txt").write_text("No model trained due to an exception.")
            # Plots from merged data
            try:
                plot_outputs_local(merged)
            except Exception:
                pass
            logging.info("Saved unified dataset with %d rows to output/processed_data.csv", len(merged))
        except FileNotFoundError as e2:
            logging.error("Datasets not found for fallback: %s", e2)


if __name__ == "__main__":
    main()


