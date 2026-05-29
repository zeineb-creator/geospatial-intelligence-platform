"""
climate_fetcher.py — NASA POWER API auto-fetch
===============================================
Automatically fetches 15 years of monthly climate data for any
coordinates using the NASA POWER API. No manual CSV export needed.

Parameters fetched:
  PRECTOTCORR : Precipitation (mm/day → converted to mm/month)
  T2M         : Temperature at 2m (°C)
  RH2M        : Relative Humidity at 2m (%)
  ALLSKY_SFC_SW_DWN : Solar radiation (optional, for PET calculation)

API docs: https://power.larc.nasa.gov/docs/
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime


NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Parameters to fetch — maps NASA code → canonical column name
PARAMETERS = {
    "PRECTOTCORR": "rainfall_mm",
    "T2M":         "temperature_c",
    "RH2M":        "humidity_pct",
}

# Days per month for rainfall conversion (mm/day → mm/month)
DAYS_PER_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}


def fetch_nasa_power(
    lat: float,
    lon: float,
    start_year: int = None,
    end_year: int = None,
    timeout: int = 30,
) -> pd.DataFrame | None:
    """
    Fetch monthly climate data from NASA POWER for a given location.

    Parameters:
        lat, lon    : Coordinates
        start_year  : First year (default: current_year - 15)
        end_year    : Last year (default: current_year - 1)
        timeout     : Request timeout in seconds

    Returns:
        DataFrame with columns: year, month, rainfall_mm, temperature_c, humidity_pct
        Returns None on failure.
    """
    current_year = datetime.now().year
    if start_year is None:
        start_year = current_year - 15
    if end_year is None:
        end_year = current_year - 1

    # Clamp to NASA POWER data availability
    start_year = max(start_year, 1981)
    end_year   = min(end_year, current_year - 1)

    params_str = ",".join(PARAMETERS.keys())

    url = (
        f"{NASA_POWER_BASE}"
        f"?parameters={params_str}"
        f"&community=AG"
        f"&longitude={lon:.4f}"
        f"&latitude={lat:.4f}"
        f"&start={start_year}"
        f"&end={end_year}"
        f"&format=JSON"
    )

    print(f"  [POWER] Fetching climate data: lat={lat:.4f}, lon={lon:.4f}, "
          f"{start_year}–{end_year}")
    print(f"  [POWER] URL: {url[:80]}...")

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        print("  [POWER] Request timed out — try again or upload CSV manually")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  [POWER] Request failed: {e}")
        return None
    except Exception as e:
        print(f"  [POWER] Unexpected error: {e}")
        return None

    # Parse response
    try:
        parameter_data = data.get("properties", {}).get("parameter", {})
        if not parameter_data:
            print("  [POWER] Empty response — coordinates may be over ocean or invalid")
            return None

        rows = []
        for nasa_param, canonical_name in PARAMETERS.items():
            if nasa_param not in parameter_data:
                print(f"  [POWER] Parameter {nasa_param} not in response")
                continue

            monthly_data = parameter_data[nasa_param]
            for yyyymm, value in monthly_data.items():
                if value == -999.0 or value is None:
                    continue
                year  = int(yyyymm[:4])
                month = int(yyyymm[4:])

                # Convert rainfall from mm/day → mm/month
                if canonical_name == "rainfall_mm":
                    value = value * DAYS_PER_MONTH.get(month, 30)

                rows.append({
                    "year":          year,
                    "month":         month,
                    canonical_name:  round(float(value), 3),
                })

        if not rows:
            print("  [POWER] No valid data rows parsed")
            return None

        # Pivot: one row per year-month with all parameters
        df = pd.DataFrame(rows)
        df = df.groupby(["year", "month"]).first().reset_index()
        df = df.sort_values(["year", "month"]).reset_index(drop=True)

        # Fill any missing canonical columns
        for col in ["rainfall_mm", "temperature_c", "humidity_pct"]:
            if col not in df.columns:
                df[col] = np.nan

        n_years  = df["year"].nunique()
        n_months = len(df)
        print(f"  [POWER] Fetched {n_months} months across {n_years} years "
              f"({start_year}–{end_year})")
        print(f"  [POWER] Rainfall range: "
              f"{df['rainfall_mm'].min():.1f}–{df['rainfall_mm'].max():.1f} mm/month")
        print(f"  [POWER] Temperature range: "
              f"{df['temperature_c'].min():.1f}–{df['temperature_c'].max():.1f} °C")

        return df

    except Exception as e:
        print(f"  [POWER] Parse error: {e}")
        return None


def fetch_annual_summary(
    lat: float,
    lon: float,
    start_year: int = None,
    end_year: int = None,
) -> pd.DataFrame | None:
    """
    Returns annual totals/means derived from monthly data.
    Useful for aridity index calculation and trend detection.
    """
    monthly = fetch_nasa_power(lat, lon, start_year, end_year)
    if monthly is None:
        return None

    annual = monthly.groupby("year").agg(
        rainfall_mm_total  = ("rainfall_mm",    "sum"),
        temperature_c_mean = ("temperature_c",  "mean"),
        humidity_pct_mean  = ("humidity_pct",   "mean"),
    ).reset_index()

    return annual


def save_to_temp_csv(df: pd.DataFrame) -> str:
    """Save DataFrame to a temp CSV file and return the path."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", mode='w'
    ) as tmp:
        df.to_csv(tmp, index=False)
        return tmp.name


def climate_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Assess quality of fetched climate data.
    Returns a dict of quality flags.
    """
    report = {
        "n_months":       len(df),
        "n_years":        df["year"].nunique() if "year" in df.columns else 0,
        "missing_rain":   df["rainfall_mm"].isna().sum() if "rainfall_mm" in df.columns else 0,
        "missing_temp":   df["temperature_c"].isna().sum() if "temperature_c" in df.columns else 0,
        "missing_hum":    df["humidity_pct"].isna().sum() if "humidity_pct" in df.columns else 0,
        "has_15yr":       False,
        "suitable":       False,
    }
    report["has_15yr"]  = report["n_years"] >= 10
    report["suitable"]  = (
        report["has_15yr"] and
        report["missing_rain"] < report["n_months"] * 0.1
    )
    return report
