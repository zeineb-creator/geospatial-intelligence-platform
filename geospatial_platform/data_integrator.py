import numpy as np
import pandas as pd
import sys
sys.path.append("/kaggle/working")

from geospatial_platform.context import InputContext


CATEGORY_KEYWORDS = {
    "rainfall":    ["rain", "rainfall", "precipitation", "precip", "prectotcorr"],
    "temperature": ["temp", "temperature", "celsius", "fahrenheit", "t2m"],
    "humidity":    ["humid", "humidity", "moisture", "rh2m"],
    "ndvi":        ["ndvi", "vegetation_index", "greenness"],
    "drought":     ["drought", "aridity", "dry"],
    "flood":       ["flood", "inundation", "overflow"],
    "wind":        ["wind", "gust", "breeze"],
    "elevation":   ["elev", "elevation", "altitude", "height"],
    "solar":       ["solar", "radiation", "irradiance", "allsky"],
}

EXCLUDED_COLUMNS = ["month", "year", "month_name", "date", "doy", "day"]


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

        # Use median and MAD — robust to seasonal outliers
        median = series.median()
        mad    = (series - median).abs().median()

        if mad == 0:
            continue

        # Check seasonality — if highly seasonal, skip anomaly flagging
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
            "category": category,
            "latest":   round(float(series.iloc[-1]), 4),
            "mean":     round(float(series.mean()), 4),
            "min":      round(float(series.min()), 4),
            "max":      round(float(series.max()), 4),
            "trend":    trend,
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

    df = context.csv_df

    category_map = {col: detect_column_category(col) for col in df.columns}
    print(f"  Columns detected: {category_map}")

    env_summary = build_environmental_summary(df, category_map)
    print(f"  Environmental variables extracted: {list(env_summary.keys())}")

    csv_anomalies = extract_anomalies_from_csv(df, category_map)
    print(f"  CSV anomalies: {csv_anomalies if csv_anomalies else 'none'}")

    all_anomalies = list(context.anomalies or []) + csv_anomalies
    context.anomalies = all_anomalies

    llm_text = format_summary_for_llm(env_summary, csv_anomalies)

    context.csv_summary = {
        "env_summary":   env_summary,
        "csv_anomalies": csv_anomalies,
        "llm_text":      llm_text,
    }

    print("=== Data integrator complete ===\n")
    return context


def build_climate_summary(df) -> dict:
    """
    Derive the climate_summary dict from the raw NASA POWER DataFrame.
    Call this in your climate processor (step 5) and store result in
    input_context.climate_summary.
 
    Expected DataFrame columns (any subset):
        rainfall_mm, temperature_c, humidity_pct
    """
    import numpy as np
 
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
        cv = s.std() / s.mean() if s.mean() != 0 else 0
        return {
            f"{col}_latest": float(s.iloc[-1]),
            f"{col}_mean":   float(s.mean()),
            f"{col}_trend":  trend,
            f"{col}_cv":     float(cv),
        }
 
    for col in ["rainfall_mm", "temperature_c", "humidity_pct"]:
        summary.update(_stats(col))
 
    return summary

def populate_convenience_fields(ic: InputContext):
    """
    After building climate_summary, populate the convenience aliases
    and derived fields on the InputContext.
    Call this at the end of your climate processor step.
    """
    if ic.climate_summary:
        ic.humidity_pct  = ic.climate_summary.get("humidity_pct_latest")
        ic.rainfall_trend = ic.climate_summary.get("rainfall_mm_trend")
 
    if ic.land_cover:
        ic.water_pct = ic.land_cover.get("water", 0.0)
 
    if ic.ndvi_mean_t1 is not None and ic.ndvi_mean_t2 is not None:
        ic.ndvi_delta = ic.ndvi_mean_t2 - ic.ndvi_mean_t1

