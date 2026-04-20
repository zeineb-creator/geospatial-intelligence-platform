
import numpy as np
import pandas as pd
import sys
sys.path.append("/kaggle/working")

from geospatial_platform.context import InputContext


# Keywords that map column names to environmental categories
CATEGORY_KEYWORDS = {
    EXCLUDED_COLUMNS = ["month", "year", "month_name", "date", "doy"]

    def extract_anomalies_from_csv(df: pd.DataFrame, category_map: dict) -> list:
        anomalies = []
        for col, category in category_map.items():
            if col.lower() in EXCLUDED_COLUMNS:
                continue
            if category == "other":
                continue
        # ... rest of function stays the same
    "rainfall":    ["rain", "rainfall", "precipitation", "precip"],
    "temperature": ["temp", "temperature", "celsius", "fahrenheit"],
    "humidity":    ["humid", "humidity", "moisture"],
    "ndvi":        ["ndvi", "vegetation_index", "greenness"],
    "drought":     ["drought", "aridity", "dry"],
    "flood":       ["flood", "inundation", "overflow"],
    "wind":        ["wind", "gust", "breeze"],
    "elevation":   ["elev", "elevation", "altitude", "height"],
}


def detect_column_category(col_name: str) -> str:
    """
    Map a column name to an environmental category using keyword matching.
    Returns 'other' if no match found.
    """
    col_lower = col_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in col_lower for kw in keywords):
            return category
    return "other"


def extract_anomalies_from_csv(df: pd.DataFrame, category_map: dict) -> list:
    """
    Detect statistical anomalies in numeric columns.
    An anomaly = a value more than 1.5 std deviations from the column mean.
    Returns a list of human-readable anomaly strings.
    """
    anomalies = []

    for col, category in category_map.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        series = df[col].dropna()
        if len(series) < 3:
            continue

        mean = series.mean()
        std  = series.std()

        if std == 0:
            continue

        latest = series.iloc[-1]
        z_score = (latest - mean) / std

        if z_score < -1.5:
            pct_change = round((latest - mean) / abs(mean) * 100, 1)
            anomalies.append(
                f"{category} anomaly: {col} is {abs(pct_change)}% below average "
                f"(latest={latest:.2f}, mean={mean:.2f})"
            )
        elif z_score > 1.5:
            pct_change = round((latest - mean) / abs(mean) * 100, 1)
            anomalies.append(
                f"{category} anomaly: {col} is {pct_change}% above average "
                f"(latest={latest:.2f}, mean={mean:.2f})"
            )

    return anomalies


def build_environmental_summary(df: pd.DataFrame, category_map: dict) -> dict:
    """
    Build a clean key-value summary of environmental variables.
    Each entry contains current value, mean, trend direction.
    """
    summary = {}

    for col, category in category_map.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        # Trend: compare last value to first value
        trend = "stable"
        if len(series) >= 2:
            delta = series.iloc[-1] - series.iloc[0]
            if delta > series.std() * 0.5:
                trend = "increasing"
            elif delta < -series.std() * 0.5:
                trend = "decreasing"

        summary[col] = {
            "category":     category,
            "latest":       round(float(series.iloc[-1]), 4),
            "mean":         round(float(series.mean()), 4),
            "min":          round(float(series.min()), 4),
            "max":          round(float(series.max()), 4),
            "trend":        trend,
            "n_records":    len(series),
        }

    return summary


def format_summary_for_llm(summary: dict, anomalies: list) -> str:
    """
    Convert the structured summary into a clean text block
    ready to be injected into the LLM prompt.
    """
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


def run_data_integrator(context: InputContext) -> InputContext:
    """
    Main entry point. Reads CSV from context, enriches context
    with structured environmental summary and CSV-derived anomalies.
    """
    print("=== Data Integrator ===")

    if context.csv_df is None:
        print("  No CSV data provided. Skipping.")
        context.csv_summary = {}
        print("=== Data integrator skipped ===\n")
        return context

    df = context.csv_df

    # Map each column to an environmental category
    category_map = {col: detect_column_category(col) for col in df.columns}
    print(f"  Columns detected: {category_map}")

    # Build structured summary
    env_summary = build_environmental_summary(df, category_map)
    print(f"  Environmental variables extracted: {list(env_summary.keys())}")

    # Detect anomalies
    csv_anomalies = extract_anomalies_from_csv(df, category_map)
    print(f"  CSV anomalies: {csv_anomalies if csv_anomalies else 'none'}")

    # Merge CSV anomalies with vision anomalies (already in context)
    all_anomalies = list(context.anomalies or []) + csv_anomalies
    context.anomalies = all_anomalies

    # Format for LLM
    llm_text = format_summary_for_llm(env_summary, csv_anomalies)

    # Store in context
    context.csv_summary = {
        "env_summary":  env_summary,
        "csv_anomalies": csv_anomalies,
        "llm_text":     llm_text,
    }

    print("=== Data integrator complete ===\n")
    return context
