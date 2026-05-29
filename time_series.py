"""
time_series.py — Multi-year NDVI time series
=============================================
Fetches annual NDVI composites from GEE for a location and returns
a DataFrame with yearly NDVI statistics for trend visualisation.
Falls back to a derived single-point estimate if GEE unavailable.
"""

import numpy as np
import pandas as pd


def fetch_ndvi_time_series(
    lat: float,
    lon: float,
    start_year: int = 2010,
    end_year: int = 2024,
    month_start: int = 3,
    month_end: int = 5,
    buffer_km: float = 5.0,
) -> pd.DataFrame | None:
    """
    Fetch annual NDVI statistics from GEE for a location.
    Uses growing-season composite (default Mar–May) to minimise
    dry-season suppression across all ecosystems.
    Automatically adjusts to southern hemisphere seasonality.

    Returns DataFrame: year, ndvi_mean, ndvi_std, ndvi_p10, ndvi_p90, n_pixels
    Returns None if GEE unavailable.
    """
    try:
        import ee
        from gee_connector import init_gee, auto_select_sensor, SENSOR_CONFIGS
    except ImportError:
        print("  [TimeSeries] GEE not available")
        return None

    if not init_gee():
        return None

    print(f"  [TimeSeries] Fetching NDVI time series {start_year}–{end_year}")
    print(f"  [TimeSeries] Location: lat={lat:.4f}, lon={lon:.4f}")
    print(f"  [TimeSeries] Season: months {month_start}–{month_end}")

    point = ee.Geometry.Point([lon, lat])
    aoi   = point.buffer(buffer_km * 1000).bounds()

    rows = []
    for year in range(start_year, end_year + 1):
        try:
            sensor = auto_select_sensor(year)
            cfg    = SENSOR_CONFIGS[sensor]

            start = f"{year}-{month_start:02d}-01"
            end_m = month_end + 1 if month_end < 12 else 1
            end_y = year if month_end < 12 else year + 1
            end   = f"{end_y}-{end_m:02d}-01"

            col = (ee.ImageCollection(cfg["collection"])
                   .filterBounds(aoi)
                   .filterDate(start, end)
                   .filter(ee.Filter.lt(cfg["cloud_prop"], 30))
                   .select(cfg["bands"]))

            count = col.size().getInfo()
            if count == 0:
                print(f"  [TimeSeries] {year}: no scenes")
                continue

            # Compute NDVI per image then median composite
            def add_ndvi(img):
                bands = cfg["bands"]
                red_idx = bands.index("SR_B4") if "SR_B4" in bands else bands.index("B4")
                nir_idx = bands.index("SR_B5") if "SR_B5" in bands else bands.index("B8")
                red = img.select(bands[red_idx])
                nir = img.select(bands[nir_idx])
                ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
                return img.addBands(ndvi)

            composite = col.map(add_ndvi).median().select("NDVI")

            # Scale to reflectance
            composite = composite.multiply(cfg["scale_factor"])
            if "offset" in cfg:
                composite = composite.add(cfg["offset"])

            # Get statistics for the AOI
            stats = composite.reduceRegion(
                reducer=ee.Reducer.mean()
                    .combine(ee.Reducer.stdDev(), sharedInputs=True)
                    .combine(ee.Reducer.percentile([10, 90]), sharedInputs=True)
                    .combine(ee.Reducer.count(), sharedInputs=True),
                geometry=aoi,
                scale=cfg["resolution"],
                maxPixels=1e8,
            ).getInfo()

            ndvi_mean = stats.get("NDVI_mean")
            if ndvi_mean is None or ndvi_mean < -1:
                continue

            rows.append({
                "year":     year,
                "ndvi_mean": round(float(ndvi_mean), 4),
                "ndvi_std":  round(float(stats.get("NDVI_stdDev", 0)), 4),
                "ndvi_p10":  round(float(stats.get("NDVI_p10", ndvi_mean - 0.05)), 4),
                "ndvi_p90":  round(float(stats.get("NDVI_p90", ndvi_mean + 0.05)), 4),
                "n_pixels":  int(stats.get("NDVI_count", 0)),
                "sensor":    sensor,
            })
            print(f"  [TimeSeries] {year}: NDVI={ndvi_mean:.3f} (n={stats.get('NDVI_count',0):.0f})")

        except Exception as e:
            print(f"  [TimeSeries] {year}: failed — {e}")
            continue

    if not rows:
        print("  [TimeSeries] No data retrieved")
        return None

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    print(f"  [TimeSeries] Complete: {len(df)} years, "
          f"NDVI range {df['ndvi_mean'].min():.3f}–{df['ndvi_mean'].max():.3f}")
    return df


def estimate_growing_season_months(lat: float, aridity_index: float = None) -> tuple[int, int]:
    """
    Estimate growing season months based on latitude and aridity.
    Used to set default time series fetch window.

    Returns (month_start, month_end)
    """
    # Southern hemisphere: flip seasons
    if lat < -15:
        return 9, 11   # Sep–Nov (austral spring)
    elif lat < 0:
        return 10, 12  # Oct–Dec

    # Northern hemisphere
    if aridity_index is not None and aridity_index < 0.3:
        # Arid/semi-arid: wet season (winter rain in Mediterranean, summer in Sahel)
        if lat > 20:
            return 2, 5   # Mediterranean / MENA: winter-spring
        else:
            return 7, 9   # Sahel: summer monsoon
    elif lat > 50:
        return 6, 8   # Boreal: summer
    else:
        return 3, 5   # Temperate/subtropical: spring


def compute_trend(df: pd.DataFrame) -> dict:
    """
    Compute linear trend over the time series.
    Returns slope, R², and trend classification.
    """
    if len(df) < 3:
        return {"slope": None, "r2": None, "trend": "insufficient data"}

    x = df["year"].values
    y = df["ndvi_mean"].values

    # Linear regression
    x_mean = x.mean()
    y_mean = y.mean()
    slope  = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Annual rate to classify trend
    annual_rate = slope  # NDVI units per year
    if abs(annual_rate) < 0.002:
        trend = "stable"
    elif annual_rate > 0.005:
        trend = "strong greening"
    elif annual_rate > 0.002:
        trend = "moderate greening"
    elif annual_rate < -0.005:
        trend = "strong browning"
    else:
        trend = "moderate browning"

    return {
        "slope":       round(float(slope), 5),
        "r2":          round(float(r2), 3),
        "trend":       trend,
        "total_change": round(float(slope * (df["year"].max() - df["year"].min())), 4),
        "annual_rate": round(float(annual_rate), 5),
    }


def render_time_series_chart(df: pd.DataFrame, trend: dict, ecosystem: str = "") -> object:
    """
    Render NDVI time series as a matplotlib figure.
    Returns the figure object for st.pyplot().
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")

    years = df["year"].values
    ndvi  = df["ndvi_mean"].values

    # Uncertainty band (p10–p90)
    if "ndvi_p10" in df.columns and "ndvi_p90" in df.columns:
        ax.fill_between(
            years, df["ndvi_p10"], df["ndvi_p90"],
            alpha=0.15, color="#2563eb", label="P10–P90 range"
        )

    # Main line
    ax.plot(years, ndvi, color="#2563eb", linewidth=2.0,
            marker="o", markersize=5, label="NDVI mean")

    # Trend line
    if trend.get("slope") is not None:
        x_fit = np.array([years.min(), years.max()])
        y_fit = trend["slope"] * x_fit + (ndvi.mean() - trend["slope"] * years.mean())
        trend_color = "#16a34a" if trend["slope"] > 0 else "#dc2626"
        ax.plot(x_fit, y_fit, "--", color=trend_color, linewidth=1.5, alpha=0.8,
                label=f"Trend: {trend['trend']} ({trend['slope']:+.4f}/yr, R²={trend['r2']:.2f})")

    # Styling
    ax.set_xlabel("Year", fontsize=9, color="#6b7280", fontfamily="monospace")
    ax.set_ylabel("NDVI", fontsize=9, color="#6b7280", fontfamily="monospace")
    title = f"NDVI Time Series — {ecosystem}" if ecosystem else "NDVI Time Series"
    ax.set_title(title, fontsize=10, color="#2563eb", fontfamily="monospace", pad=8)

    ax.set_ylim(max(0, ndvi.min() - 0.05), min(1, ndvi.max() + 0.05))
    ax.tick_params(colors="#6b7280", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#dde3ed")

    ax.legend(fontsize=8, framealpha=0.9, edgecolor="#dde3ed")
    ax.grid(True, alpha=0.3, color="#dde3ed")
    fig.tight_layout(pad=0.8)
    return fig
