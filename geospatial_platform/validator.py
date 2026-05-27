"""
validator.py
────────────
Fix 5: has_temporal_ndvi now checks ic.ndvi_delta (set by app.py temporal block)
instead of ic.ndvi_previous (set only by main.py, which is never called from
Streamlit). Previously the validator always reported "Temporal NDVI: No" even
when two images were uploaded and the delta was correctly computed.
"""

import numpy as np
import pandas as pd

from geospatial_platform.context import InputContext


def validate_inputs(context: InputContext) -> dict:
    """
    Assess data quality and availability.
    Returns a flags dict controlling what the system can and cannot conclude.
    """
    # Fix 5: use ndvi_delta as the temporal availability signal.
    # ndvi_previous is only populated by main.py (unused in Streamlit).
    # ndvi_delta is populated by app.py after processing the second image.
    has_temporal = (
        getattr(context, 'ndvi_delta', None) is not None or
        getattr(context, 'ndvi_previous', None) is not None
    )

    flags = {
        "has_ndvi":              context.ndvi is not None,
        "has_ndwi":              context.ndwi is not None,
        "has_ndbi":              context.ndbi is not None,
        "has_temporal_ndvi":     has_temporal,
        "has_climate_data":      context.csv_df is not None,
        "has_multiyear_climate": False,
        "climate_years":         0,
        "has_rainfall":          False,
        "has_temperature":       False,
        "has_humidity":          False,
        "high_seasonality":      False,
        "flood_detectable":      False,
        "single_image_only":     not has_temporal,
        "can_detect_change":     has_temporal,
        "can_assess_drought":    False,
        "can_assess_flood":      False,
    }

    if context.csv_df is not None:
        df = context.csv_df

        year_col = next((c for c in df.columns if c.lower() == "year"), None)
        if year_col:
            years = df[year_col].nunique()
            flags["climate_years"]         = int(years)
            flags["has_multiyear_climate"] = years > 1
        else:
            flags["climate_years"]         = 1
            flags["has_multiyear_climate"] = False

        cols = [c.lower() for c in df.columns]
        flags["has_rainfall"]    = any(
            kw in c for c in cols
            for kw in ["rain", "prec", "prectotcorr", "rainfall_mm"]
        )
        flags["has_temperature"] = any(
            kw in c for c in cols
            for kw in ["temp", "t2m", "temperature_c"]
        )
        flags["has_humidity"]    = any(
            kw in c for c in cols
            for kw in ["humid", "rh2m", "qv2m", "humidity_pct"]
        )

        rain_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ["rain", "prec", "prectotcorr", "rainfall_mm"])
        ]
        if rain_cols:
            series = df[rain_cols[0]].dropna()
            if len(series) >= 3 and series.mean() > 0:
                flags["high_seasonality"] = (series.std() / series.mean()) > 0.8

    # Flood detectable: needs NDWI and some water signal
    water_ratio = getattr(context, 'water_ratio', None)
    water_pct   = (context.land_cover or {}).get("water", 0)
    flags["flood_detectable"]   = (
        flags["has_ndwi"] and
        (water_ratio is not None and water_ratio > 0.15 or water_pct > 15)
    )
    flags["can_assess_drought"] = flags["has_ndvi"] and flags["has_rainfall"]

    return flags


def build_reliability_report(flags: dict) -> str:
    """
    Build a human-readable data reliability summary for the UI and LLM prompt.
    """
    lines = ["DATA RELIABILITY ASSESSMENT:"]

    lines.append("\n  Imagery:")
    lines.append(f"    NDVI available   : {'Yes' if flags['has_ndvi'] else 'No'}")
    lines.append(f"    NDWI available   : {'Yes' if flags['has_ndwi'] else 'No'}")
    lines.append(f"    NDBI available   : {'Yes' if flags['has_ndbi'] else 'No'}")
    lines.append(
        f"    Temporal NDVI    : "
        f"{'Yes — delta computed from two images' if flags['has_temporal_ndvi'] else 'No — single image only'}"
    )

    lines.append("\n  Climate data:")
    if flags["has_climate_data"]:
        lines.append(f"    Years available  : {flags['climate_years']}")
        lines.append(
            f"    Multi-year       : "
            f"{'Yes' if flags['has_multiyear_climate'] else 'No — trend analysis limited'}"
        )
        lines.append(f"    Rainfall         : {'Yes' if flags['has_rainfall'] else 'No'}")
        lines.append(f"    Temperature      : {'Yes' if flags['has_temperature'] else 'No'}")
        lines.append(f"    Humidity         : {'Yes' if flags['has_humidity'] else 'No'}")
        lines.append(f"    High seasonality : {'Yes' if flags['high_seasonality'] else 'No'}")
    else:
        lines.append("    No climate data provided")

    lines.append("\n  Analysis capabilities:")
    lines.append(
        f"    Vegetation change: "
        f"{'Enabled — ΔNDVI computed' if flags['can_detect_change'] else 'Disabled — single image only'}"
    )
    lines.append(
        f"    Drought assessment: "
        f"{'Enabled' if flags['can_assess_drought'] else 'Partial — missing rainfall or NDVI'}"
    )
    lines.append(
        f"    Flood detection  : "
        f"{'Enabled' if flags['flood_detectable'] else 'Limited — low water signal'}"
    )

    return "\n".join(lines)
