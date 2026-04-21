import streamlit as st
import tempfile
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geospatial_platform.vision_model import load_vit
from geospatial_platform.llm_engine import load_llm
from main import run_pipeline


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Geospatial Intelligence Platform",
    page_icon="🛰",
    layout="wide",
)

# ── Model loading (cached — runs once per session) ────────────
@st.cache_resource(show_spinner="Loading vision model...")
def get_vit():
    return load_vit()

@st.cache_resource(show_spinner="Initializing language model...")
def get_llm():
    groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    os.environ["GROQ_API_KEY"] = groq_key
    from geospatial_platform.llm_engine import load_llm
    return load_llm()


# ── Helper: index map ────────────────────────────────────────
def plot_index_map(array: np.ndarray, title: str, cmap: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(array, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ── Helper: land cover bar chart ─────────────────────────────
def plot_land_cover(land_cover: dict) -> plt.Figure:
    labels = [k for k, v in land_cover.items() if v > 0]
    values = [v for k, v in land_cover.items() if v > 0]
    if not labels:
        return None
    colors = ["#3b8bd4", "#2d9e5f", "#c0623d", "#b4a98a", "#888780"][:len(labels)]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(labels, values, color=colors, height=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_xlabel("Coverage (%)")
    ax.set_title("Land cover estimate", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── Helper: RGB preview ───────────────────────────────────────
def plot_rgb(image_array: np.ndarray) -> plt.Figure:
    rgb = np.transpose(image_array[:3], (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.set_title("RGB composite", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ── Helper: climate time series ──────────────────────────────
def plot_climate_timeseries(csv_df) -> plt.Figure | None:
    if csv_df is None:
        return None
    rain_cols = [c for c in csv_df.columns if "rain" in c.lower() or "prec" in c.lower()]
    temp_cols = [c for c in csv_df.columns if "temp" in c.lower()]
    if not rain_cols and not temp_cols:
        return None

    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    x = range(len(csv_df))

    if "month_name" in csv_df.columns:
        labels = csv_df["month_name"].tolist()
    else:
        labels = list(x)

    if rain_cols:
        ax1.bar(x, csv_df[rain_cols[0]], color="#3b8bd4", alpha=0.6, label="Rainfall (mm)")
        ax1.set_ylabel("Rainfall (mm)", color="#3b8bd4")
        ax1.tick_params(axis="y", labelcolor="#3b8bd4")

    if temp_cols:
        ax2 = ax1.twinx()
        ax2.plot(x, csv_df[temp_cols[0]], color="#c0623d",
                 linewidth=2, marker="o", markersize=4, label="Temperature (°C)")
        ax2.set_ylabel("Temperature (°C)", color="#c0623d")
        ax2.tick_params(axis="y", labelcolor="#c0623d")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_title("Climate — monthly rainfall & temperature", fontsize=11)
    ax1.spines[["top"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── Helper: NDVI histogram ────────────────────────────────────
def plot_ndvi_histogram(ndvi: np.ndarray) -> plt.Figure | None:
    if ndvi is None:
        return None
    flat = ndvi.flatten()
    flat = flat[(flat > -1) & (flat < 1) & ~np.isnan(flat)]  # ← remove NaN
    if len(flat) == 0:
        return None

    mean_val = float(np.nanmean(ndvi[~np.isnan(ndvi)])) if not np.all(np.isnan(ndvi)) else 0

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(flat, bins=60, color="#2d9e5f", alpha=0.75, edgecolor="none")
    ax.axvline(x=mean_val, color="#c0623d", linewidth=1.5,
               linestyle="--", label=f"Mean={mean_val:.3f}")
    ax.axvline(x=0.1,  color="#888780", linewidth=1, linestyle=":",  label="Sparse (0.10)")
    ax.axvline(x=0.25, color="#888780", linewidth=1, linestyle="-.", label="Moderate (0.25)")
    ax.axvline(x=0.45, color="#444441", linewidth=1, linestyle="-.", label="Dense (0.45)")
    ax.set_xlabel("NDVI value")
    ax.set_ylabel("Pixel count")
    ax.set_title("NDVI distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── Helper: NDVI change map ───────────────────────────────────
def plot_ndvi_change_map(trend_map: np.ndarray, trend_val: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(trend_map, cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    direction = "improvement" if trend_val > 0 else "decline"
    ax.set_title(
        f"ΔNDVI change map (mean={trend_val:+.3f})\n"
        f"Green = {direction}   Red = opposite",
        fontsize=10
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    return fig


# ── Helper: full report text ──────────────────────────────────
def build_full_interpretation(results: dict) -> str:
    lines = []

    lines.append("=" * 60)
    lines.append("GEOSPATIAL INTELLIGENCE REPORT")
    lines.append("=" * 60)
    lines.append(f"\nQuestion: {results['user_question']}\n")

    lines.append("── IMAGE METADATA ──────────────────────────────────────")
    meta = results["image_meta"]
    lines.append(f"  Format     : {results['image_format'].upper()}")
    lines.append(f"  Bands      : {results['n_bands']}")
    lines.append(f"  Resolution : {meta.get('resolution', 'unknown')}")
    lines.append(f"  CRS        : {meta.get('crs', 'unknown')}")
    lines.append(f"  Dimensions : {meta.get('width')} x {meta.get('height')} px")
    lines.append(f"  Context    : {meta.get('region_context', 'unknown')}")
    lines.append(f"  Region     : {results.get('region_name', 'Unknown')}")
    lines.append(f"  Ecosystem  : {results.get('ecosystem', 'Unknown')}\n")

    lines.append("── LAND COVER ───────────────────────────────────────────")
    if results["land_cover"]:
        for cls, pct in results["land_cover"].items():
            lines.append(f"  {cls:<15}: {pct:.2f}%")
    lines.append("")

    if results.get("vegetation_breakdown"):
        lines.append("── VEGETATION BREAKDOWN ─────────────────────────────────")
        vb = results["vegetation_breakdown"]
        lines.append(f"  Sparse   (NDVI 0.10–0.25) : {vb.get('sparse_pct', 0):.2f}%")
        lines.append(f"  Moderate (NDVI 0.25–0.45) : {vb.get('moderate_pct', 0):.2f}%")
        lines.append(f"  Dense    (NDVI > 0.45)    : {vb.get('dense_pct', 0):.2f}%")
        lines.append("")

    if results.get("ndvi_trend") is not None:
        import math
        trend = results["ndvi_trend"]
        if not math.isnan(trend):
            lines.append("── TEMPORAL NDVI ANALYSIS ───────────────────────────────")
            direction = "improvement" if trend > 0 else "decline"
            lines.append(f"  ΔNDVI (mean) : {trend:+.4f} → {direction}")
            lines.append("")

    if results.get("water_ratio") is not None:
        lines.append("── HYDROLOGICAL INDICATORS ──────────────────────────────")
        lines.append(f"  Water/flood coverage : {results['water_ratio']*100:.2f}%")
        if results["water_ratio"] > 0.25:
            lines.append("  ⚠ Possible flooding detected")
        lines.append("")

    if results.get("aridity_index") is not None:
        lines.append("── ARIDITY INDEX ────────────────────────────────────────")
        ai = results["aridity_index"]
        if ai < 0.05:   label = "Hyper-arid"
        elif ai < 0.2:  label = "Arid"
        elif ai < 0.5:  label = "Semi-arid"
        elif ai < 0.65: label = "Dry sub-humid"
        else:           label = "Humid"
        lines.append(f"  Index : {ai:.3f} → {label}")
        lines.append("")

    lines.append("── DETECTED ANOMALIES ───────────────────────────────────")
    if results["anomalies"]:
        for a in results["anomalies"]:
            lines.append(f"  • {a}")
    else:
        lines.append("  No anomalies detected.")
    lines.append("")

    if results.get("confidence_score") is not None:
        lines.append("── CONFIDENCE SCORE ─────────────────────────────────────")
        score = results["confidence_score"]
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(f"  {bar} {score*100:.1f}%")
        lines.append("")

    lines.append("── RETRIEVED ENVIRONMENTAL CONTEXT ─────────────────────")
    lines.append(results["retrieved_context"] or "  No environmental data provided.")
    lines.append("")

    lines.append("── AI SCIENTIFIC REPORT ─────────────────────────────────")
    lines.append(results["final_report"] or "  No report generated.")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# UI LAYOUT
# ════════════════════════════════════════════════════════════

st.title("🛰 Geospatial Intelligence Platform")
st.caption("Upload a satellite image and optional climate data to generate a scientific interpretation.")

# ── Sidebar: inputs ──────────────────────────────────────────
with st.sidebar:
    st.header("Inputs")

    image_file = st.file_uploader(
        "Satellite image (current)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="GeoTIFF (multispectral) or RGB image"
    )

    image_file_past = st.file_uploader(
        "Past satellite image (optional)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="Upload an older image of the same area to enable temporal NDVI analysis"
    )

    csv_file = st.file_uploader(
        "Environmental data (optional)",
        type=["csv"],
        help="CSV with climate, rainfall, temperature, or sensor data"
    )

    question = st.text_area(
        "Your question",
        placeholder="e.g. Is there drought risk? What is the vegetation health status?",
        height=100,
    )

    run_button = st.button("Run analysis", type="primary", use_container_width=True)

    st.divider()

    if st.button("Clear cache & restart", type="secondary", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.caption("Models load once and stay cached for the session.")


# ── Main area: results ───────────────────────────────────────
if run_button:
    if image_file is None:
        st.error("Please upload a satellite image to continue.")
        st.stop()

    # Save current image
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(image_file.name)[-1]
    ) as tmp_img:
        tmp_img.write(image_file.read())
        image_path = tmp_img.name

    # Save past image if provided
    image_path_past = None
    if image_file_past:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(image_file_past.name)[-1]
        ) as tmp_past:
            tmp_past.write(image_file_past.read())
            image_path_past = tmp_past.name

    # Save CSV if provided
    csv_path = None
    if csv_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            tmp_csv.write(csv_file.read())
            csv_path = tmp_csv.name

    # Load models
    extractor, vit_model = get_vit()
    tokenizer, llm       = get_llm()

    # Run pipeline
    with st.spinner("Running analysis — this takes ~30 seconds..."):
        try:
            results, extractor, vit_model = run_pipeline(
                image_path=image_path,
                csv_path=csv_path,
                question=question or None,
                image_path_past=image_path_past,
                extractor=extractor,
                vit_model=vit_model,
                tokenizer=tokenizer,
                llm=llm,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    st.success("Analysis complete.")

    # ── Region + ecosystem banner ─────────────────────────────
    region    = results.get("region_name", "Unknown region")
    ecosystem = results.get("ecosystem", "Unknown")
    col_r, col_e = st.columns(2)
    col_r.metric("Detected region", region)
    col_e.metric("Ecosystem type", ecosystem)

    # ── Row 1: image preview + land cover ────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image preview")
        fig_rgb = plot_rgb(results["image_array"])
        st.pyplot(fig_rgb)
        plt.close(fig_rgb)

    with col2:
        st.subheader("Land cover")
        fig_lc = plot_land_cover(results["land_cover"])
        if fig_lc:
            st.pyplot(fig_lc)
            plt.close(fig_lc)
        else:
            st.info("Land cover not available for this image.")

    # ── Row 2: spectral indices (3 maps) ─────────────────────
    if any(results.get(k) is not None for k in ["ndvi", "ndwi", "ndbi"]):
        st.subheader("Spectral indices")
        idx_cols = st.columns(3)

        if results.get("ndvi") is not None:
            with idx_cols[0]:
                fig_ndvi = plot_index_map(results["ndvi"], "NDVI (vegetation)", "RdYlGn")
                st.pyplot(fig_ndvi)
                plt.close(fig_ndvi)

        if results.get("ndwi") is not None:
            with idx_cols[1]:
                fig_ndwi = plot_index_map(results["ndwi"], "NDWI (water)", "Blues")
                st.pyplot(fig_ndwi)
                plt.close(fig_ndwi)

        if results.get("ndbi") is not None:
            with idx_cols[2]:
                fig_ndbi = plot_index_map(results["ndbi"], "NDBI (urban)", "OrRd")
                st.pyplot(fig_ndbi)
                plt.close(fig_ndbi)

    # ── Row 3: temporal NDVI change map ──────────────────────
    if results.get("ndvi_trend_map") is not None:
        st.subheader("NDVI change map (before → after)")
        fig_trend = plot_ndvi_change_map(
            results["ndvi_trend_map"],
            results.get("ndvi_trend", 0)
        )
        st.pyplot(fig_trend)
        plt.close(fig_trend)

    # ── Row 4: NDVI histogram ─────────────────────────────────
    if results.get("ndvi") is not None:
        st.subheader("NDVI distribution")
        fig_hist = plot_ndvi_histogram(results["ndvi"])
        if fig_hist:
            st.pyplot(fig_hist)
            plt.close(fig_hist)

    # ── Row 5: climate time series ────────────────────────────
    if results.get("csv_df") is not None:
        st.subheader("Climate time series")
        fig_climate = plot_climate_timeseries(results["csv_df"])
        if fig_climate:
            st.pyplot(fig_climate)
            plt.close(fig_climate)

    # ── Row 6: anomalies ──────────────────────────────────────
    if results["anomalies"]:
        st.subheader("Detected anomalies")
        for a in results["anomalies"]:
            st.warning(f"⚠ {a}")

    # ── Warning banners ───────────────────────────────────────
    flags = results.get("image_meta", {}).get("validator_flags", {})
    if flags.get("single_image_only"):
        st.warning("⚠ Single image only — vegetation change cannot be assessed. Upload a past image to enable temporal analysis.")
    if not flags.get("has_multiyear_climate"):
        st.warning("⚠ Single-year climate data — long-term trend analysis is limited.")
    if flags.get("high_seasonality"):
        st.info("ℹ Strong seasonal rainfall pattern detected — monthly anomalies should be interpreted in seasonal context.")
    if flags.get("flood_detectable"):
        st.error("🌊 Elevated water signal detected — possible flooding or waterlogging.")

    # ── Reliability panel ─────────────────────────────────────
    reliability = results.get("image_meta", {}).get("reliability_text", "")
    if reliability:
        with st.expander("Data reliability assessment"):
            st.code(reliability)

    # ── Full interpretation ────────────────────────────────────
    st.subheader("Full interpretation")
    full_text = build_full_interpretation(results)
    st.text_area(
        label="report",
        value=full_text,
        height=500,
        label_visibility="collapsed",
    )

    # ── Download button ────────────────────────────────────────
    st.download_button(
        label="Download report (.txt)",
        data=full_text,
        file_name="geospatial_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # Cleanup temp files
    os.unlink(image_path)
    if image_path_past:
        os.unlink(image_path_past)
    if csv_path:
        os.unlink(csv_path)

else:
    st.info("Upload an image and click **Run analysis** to get started.")
