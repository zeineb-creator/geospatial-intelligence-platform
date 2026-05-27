"""
data_integrator.py
──────────────────
Fix 4: Added NASA POWER column aliasing.
build_climate_summary() was silently returning an empty dict when the CSV
used NASA POWER native column names (PRECTOTCORR, T2M, RH2M) instead of
the expected aliases (rainfall_mm, temperature_c, humidity_pct).
Now resolves both naming conventions before computing stats.
"""

import numpy as np
import pandas as pd

from geospatial_platform.context import InputContext


# ── NASA POWER column name aliases ────────────────────────────────────────────
# Maps any known variant → canonical name used by build_climate_summary()
NASA_POWER_ALIASES = {
    # Rainfall / precipitation
    "prectotcorr":    "rainfall_mm",
    "prectot":        "rainfall_mm",
    "precipitation":  "rainfall_mm",
    "precip":         "rainfall_mm",
    "rain":           "rainfall_mm",
    "rainfall":       "rainfall_mm",
    "rainfall_mm":    "rainfall_mm",
    # Temperature
    "t2m":            "temperature_c",
    "t2m_max":        "temperature_c",
    "t2m_min":        "temperature_c",
    "temperature":    "temperature_c",
    "temp":           "temperature_c",
    "temperature_c":  "temperature_c",
    # Humidity
    "rh2m":           "humidity_pct",
    "qv2m":           "humidity_pct",
    "humidity":       "humidity_pct",
    "humidity_pct":   "humidity_pct",
    "rh":             "humidity_pct",
}

CATEGORY_KEYWORDS = {
    "rainfall":    ["rain", "rainfall", "precipitation", "precip", "prectotcorr", "prectot"],
    "temperature": ["temp", "temperature", "celsius", "fahrenheit", "t2m"],
    "humidity":    ["humid", "humidity", "moisture", "rh2m", "qv2m", "rh"],
    "ndvi":        ["ndvi", "vegetation_index", "greenness"],
    "drought":     ["drought", "aridity", "dry"],
    "flood":       ["flood", "inundation", "overflow"],
    "wind":        ["wind", "gust", "breeze"],
    "elevation":   ["elev", "elevation", "altitude", "height"],
    "solar":       ["solar", "radiation", "irradiance", "allsky"],
}

EXCLUDED_COLUMNS = ["month", "year", "month_name", "date", "doy", "day"]


def resolve_nasa_power_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename any NASA POWER native column names to canonical aliases.
    Returns a copy of the DataFrame with renamed columns where applicable.
    Warns when a rename occurs so the user can see what was resolved.
    """
    rename_map = {}
    for col in df.columns:
        canonical = NASA_POWER_ALIASES.get(col.lower())
        if canonical and col != canonical and canonical not in df.columns:
            rename_map[col] = canonical
            print(f"  [Alias] {col} → {canonical}")

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def detect_column_category(col_name: str) -> str:
    col_lower = col_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in col_lower for kw in keywords):
            return category
    return "other"


def extract_anomalies_from_csv(df: pd.DataFrame, category_map: dict) -> list:
    anomalies = []

    for col, category in category_map.items():
        if col.lower() in EXCLUDED_COLUMNS:
            continue
        if category == "other":
            continue
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        series = df[col].dropna()
        if len(series) < 3:
            continue

        median = series.median()
        mad    = (series - median).abs().median()
        if mad == 0:
            continue

        if median > 0:
            std_ratio = series.std() / series.mean()
            if std_ratio > 0.8 and category == "rainfall":
                anomalies.append(
                    f"{category} note: {col} shows strong seasonality "
                    f"(CV={std_ratio:.2f}) — monthly comparisons may be misleading"
                )
                continue

        latest  = series.iloc[-1]
        z_score = (latest - median) / (mad * 1.4826 + 1e-8)

        if z_score < -2.5:
            pct = round((latest - median) / abs(median) * 100, 1)
            anomalies.append(
                f"{category} deficit: {col} is {abs(pct):.1f}% below seasonal median "
                f"(latest={latest:.2f}, median={median:.2f})"
            )
        elif z_score > 2.5:
            pct = round((latest - median) / abs(median) * 100, 1)
            anomalies.append(
                f"{category} surplus: {col} is {pct:.1f}% above seasonal median "
                f"(latest={latest:.2f}, median={median:.2f})"
            )

    return anomalies


def build_environmental_summary(df: pd.DataFrame, category_map: dict) -> dict:
    summary = {}

    for col, category in category_map.items():
        if col.lower() in EXCLUDED_COLUMNS:
            continue
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        trend = "stable"
        if len(series) >= 2:
            delta = series.iloc[-1] - series.iloc[0]
            if delta > series.std() * 0.5:
                trend = "increasing"
            elif delta < -series.std() * 0.5:
                trend = "decreasing"

        summary[col] = {
            "category":  category,
            "latest":    round(float(series.iloc[-1]), 4),
            "mean":      round(float(series.mean()), 4),
            "min":       round(float(series.min()), 4),
            "max":       round(float(series.max()), 4),
            "trend":     trend,
            "n_records": len(series),
        }

    return summary


def format_summary_for_llm(summary: dict, anomalies: list) -> str:
    lines = ["Environmental data summary:"]
    for col, stats in summary.items():
        lines.append(
            f"  - {col} ({stats['category']}): "
            f"latest={stats['latest']}, mean={stats['mean']}, "
            f"trend={stats['trend']}"
        )
    if anomalies:
        lines.append("\nDetected anomalies:")
        for a in anomalies:
            lines.append(f"  - {a}")
    else:
        lines.append("\nNo significant anomalies detected in CSV data.")
    return "\n".join(lines)


def integrate_data(context: InputContext) -> InputContext:
    print("=== Data Integrator ===")

    if context.csv_df is None:
        print("  No CSV data provided. Skipping.")
        context.csv_summary = {}
        print("=== Data integrator skipped ===\n")
        return context

    # Fix 4: resolve NASA POWER column names before any processing
    df = resolve_nasa_power_columns(context.csv_df)
    context.csv_df = df  # store resolved version back

    category_map = {col: detect_column_category(col) for col in df.columns}
    print(f"  Columns detected: {category_map}")

    env_summary   = build_environmental_summary(df, category_map)
    print(f"  Environmental variables extracted: {list(env_summary.keys())}")

    csv_anomalies = extract_anomalies_from_csv(df, category_map)
    print(f"  CSV anomalies: {csv_anomalies if csv_anomalies else 'none'}")

    context.anomalies = list(context.anomalies or []) + csv_anomalies
    llm_text          = format_summary_for_llm(env_summary, csv_anomalies)

    context.csv_summary = {
        "env_summary":   env_summary,
        "csv_anomalies": csv_anomalies,
        "llm_text":      llm_text,
    }

    print("=== Data integrator complete ===\n")
    return context


def build_climate_summary(df: pd.DataFrame) -> dict:
    """
    Derive the climate_summary dict from a NASA POWER DataFrame.
    Fix 4: resolves column aliases before lookup so PRECTOTCORR, T2M, RH2M
    all map correctly to the canonical rainfall_mm / temperature_c / humidity_pct keys.
    """
    # Apply aliases on a working copy so we don't mutate the caller's DataFrame
    df = resolve_nasa_power_columns(df.copy())

    summary = {}

    def _stats(col):
        if col not in df.columns:
            return {}
        s = df[col].dropna()
        if s.empty:
            return {}
        trend = "stable"
        if len(s) > 2:
            half = len(s) // 2
            if s.iloc[-half:].mean() > s.iloc[:half].mean() * 1.1:
                trend = "increasing"
            elif s.iloc[-half:].mean() < s.iloc[:half].mean() * 0.9:
                trend = "decreasing"
        mean = s.mean()
        cv   = float(s.std() / mean) if mean != 0 else 0.0
        return {
            f"{col}_latest": float(s.iloc[-1]),
            f"{col}_mean":   float(mean),
            f"{col}_trend":  trend,
            f"{col}_cv":     cv,
        }

    for col in ["rainfall_mm", "temperature_c", "humidity_pct"]:
        result = _stats(col)
        if result:
            summary.update(result)
        else:
            print(f"  [Climate] column '{col}' not found or empty after aliasing")

    if not summary:
        print("  [Climate] WARNING: build_climate_summary returned empty — "
              "check CSV column names vs NASA_POWER_ALIASES")

    return summary


def populate_convenience_fields(ic: InputContext) -> None:
    """
    Populate convenience aliases on InputContext after climate_summary is built.
    Fix 5 (partial): also sets water_pct from land_cover.
    """
    if ic.climate_summary:
        ic.humidity_pct   = ic.climate_summary.get("humidity_pct_latest")
        ic.rainfall_trend = ic.climate_summary.get("rainfall_mm_trend")

    if ic.land_cover:
        ic.water_pct = ic.land_cover.get("water", 0.0)

    # ndvi_delta is set by app.py temporal block; don't overwrite if already set
    if not getattr(ic, 'ndvi_delta', None):
        if getattr(ic, 'ndvi_mean_t1', None) is not None and \
           getattr(ic, 'ndvi_mean_t2', None) is not None:
            ic.ndvi_delta = round(ic.ndvi_mean_t2 - ic.ndvi_mean_t1, 4)
