
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
    import os
    groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    os.environ["GROQ_API_KEY"] = groq_key
    from geospatial_platform.llm_engine import load_llm
    return load_llm()

# ── Helper: index map figure ──────────────────────────────────
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

    colors = ["#2d9e5f", "#3b8bd4", "#c0623d", "#b4a98a", "#888780"][:len(labels)]
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

# ── Helper: build full interpretation text ───────────────────
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
    lines.append(f"  Dimensions : {meta.get('width')} x {meta.get('height')} px\n")

    lines.append("── LAND COVER ───────────────────────────────────────────")
    if results["land_cover"]:
        for cls, pct in results["land_cover"].items():
            if pct > 0:
                lines.append(f"  {cls:<15}: {pct:.2f}%")
    else:
        lines.append("  Not available.")
    lines.append("")

    lines.append("── DETECTED ANOMALIES ───────────────────────────────────")
    if results["anomalies"]:
        for a in results["anomalies"]:
            lines.append(f"  • {a}")
    else:
        lines.append("  No anomalies detected.")
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
        "Satellite image",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="GeoTIFF (multispectral) or RGB image"
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
    st.caption("Models load once and stay cached for the session.")


# ── Main area: results ───────────────────────────────────────
if run_button:
    if image_file is None:
        st.error("Please upload a satellite image to continue.")
        st.stop()

    # Save uploaded files to temp paths
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(image_file.name)[-1]
    ) as tmp_img:
        tmp_img.write(image_file.read())
        image_path = tmp_img.name

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
                extractor=extractor,
                vit_model=vit_model,
                tokenizer=tokenizer,
                llm=llm,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    st.success("Analysis complete.")

    # ── Row 1: image preview + land cover chart ───────────────
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

    # ── Row 2: index maps ─────────────────────────────────────
    if results["ndvi"] is not None or results["ndwi"] is not None:
        st.subheader("Spectral indices")
        idx_cols = st.columns(2)

        if results["ndvi"] is not None:
            with idx_cols[0]:
                fig_ndvi = plot_index_map(results["ndvi"], "NDVI (vegetation)", "RdYlGn")
                st.pyplot(fig_ndvi)
                plt.close(fig_ndvi)

        if results["ndwi"] is not None:
            with idx_cols[1]:
                fig_ndwi = plot_index_map(results["ndwi"], "NDWI (water)", "Blues")
                st.pyplot(fig_ndwi)
                plt.close(fig_ndwi)

    # ── Row 3: anomalies ──────────────────────────────────────
    if results["anomalies"]:
        st.subheader("Detected anomalies")
        for a in results["anomalies"]:
            st.warning(f"⚠ {a}")

    # ── Row 4: full interpretation ────────────────────────────
    st.subheader("Full interpretation")
    full_text = build_full_interpretation(results)
    st.text_area(
        label="",
        value=full_text,
        height=500,
        label_visibility="collapsed",
    )

    # ── Row 5: download button ────────────────────────────────
    st.download_button(
        label="Download report (.txt)",
        data=full_text,
        file_name="geospatial_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # Cleanup temp files
    os.unlink(image_path)
    if csv_path:
        os.unlink(csv_path)

else:
    st.info("Upload an image and click **Run analysis** to get started.")
