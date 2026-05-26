import numpy as np
import pandas as pd

from geospatial_platform.context import InputContext


def validate_inputs(context: InputContext) -> dict:
    """
    Assess data quality and availability.
    Returns a flags dict that controls what the system can and cannot conclude.
    """
    flags = {
        "has_ndvi":              context.ndvi is not None,
        "has_ndwi":              context.ndwi is not None,
        "has_ndbi":              context.ndbi is not None,
        "has_temporal_ndvi":     context.ndvi_previous is not None,
        "has_climate_data":      context.csv_df is not None,
        "has_multiyear_climate": False,
        "climate_years":         0,
        "has_rainfall":          False,
        "has_temperature":       False,
        "has_humidity":          False,
        "high_seasonality":      False,
        "flood_detectable":      False,
        "single_image_only":     context.ndvi_previous is None,
        "can_detect_change":     False,
        "can_assess_drought":    False,
        "can_assess_flood":      False,
    }

    if context.csv_df is not None:
        df = context.csv_df

        # Check years — handle both 'year' and 'YEAR' column names
        year_col = next((c for c in df.columns if c.lower() == "year"), None)
        if year_col:
            years = df[year_col].nunique()
            flags["climate_years"]         = int(years)
            flags["has_multiyear_climate"] = years > 1
        else:
            flags["climate_years"]         = 1
            flags["has_multiyear_climate"] = False

        # Check parameter coverage
        cols = [c.lower() for c in df.columns]
        flags["has_rainfall"]    = any("rain" in c or "prec" in c for c in cols)
        flags["has_temperature"] = any("temp" in c or "t2m" in c for c in cols)
        flags["has_humidity"]    = any("humid" in c or "rh" in c for c in cols)

        # Check seasonality
        rain_cols = [c for c in df.columns if "rain" in c.lower() or "prec" in c.lower()]
        if rain_cols:
            series = df[rain_cols[0]].dropna()
            if len(series) >= 3 and series.mean() > 0:
                std_ratio = series.std() / series.mean()
                flags["high_seasonality"] = std_ratio > 0.8

    # Derived capability flags
    flags["can_detect_change"]  = flags["has_temporal_ndvi"]
    flags["flood_detectable"]   = (flags["has_ndwi"] and
                                   context.water_ratio is not None and
                                   context.water_ratio > 0.15)
    flags["can_assess_drought"] = (flags["has_ndvi"] and flags["has_rainfall"])

    return flags


def build_reliability_report(flags: dict) -> str:
    """
    Build a human-readable data reliability summary
    to inject into the LLM prompt and display in the UI.
    """
    lines = ["DATA RELIABILITY ASSESSMENT:"]

    lines.append("\n  Imagery:")
    lines.append(f"    NDVI available        : {'Yes' if flags['has_ndvi'] else 'No'}")
    lines.append(f"    NDWI available        : {'Yes' if flags['has_ndwi'] else 'No'}")
    lines.append(f"    NDBI available        : {'Yes' if flags['has_ndbi'] else 'No'}")
    lines.append(f"    Temporal NDVI         : {'Yes' if flags['has_temporal_ndvi'] else 'No — single image only'}")

    lines.append("\n  Climate data:")
    if flags["has_climate_data"]:
        lines.append(f"    Years available       : {flags['climate_years']}")
        lines.append(f"    Multi-year            : {'Yes' if flags['has_multiyear_climate'] else 'No — trend analysis limited'}")
        lines.append(f"    Rainfall              : {'Yes' if flags['has_rainfall'] else 'No'}")
        lines.append(f"    Temperature           : {'Yes' if flags['has_temperature'] else 'No'}")
        lines.append(f"    High seasonality      : {'Yes' if flags['high_seasonality'] else 'No'}")
    else:
        lines.append("    No climate data provided")

    lines.append("\n  Analysis capabilities:")
    lines.append(f"    Vegetation change     : {'Enabled' if flags['can_detect_change'] else 'Disabled — no temporal data'}")
    lines.append(f"    Drought assessment    : {'Enabled' if flags['can_assess_drought'] else 'Partial — missing rainfall or NDVI'}")
    lines.append(f"    Flood detection       : {'Enabled' if flags['flood_detectable'] else 'Limited — low water signal'}")

    return "\n".join(lines)
