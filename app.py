import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import io
import sys
import os

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Geospatial Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #58a6ff !important;
    letter-spacing: -0.02em;
}
h1 { font-size: 1.6rem !important; }
h2 { font-size: 1.15rem !important; border-bottom: 1px solid #21262d; padding-bottom: 0.4rem; }
h3 { font-size: 0.95rem !important; color: #79c0ff !important; }

/* Cards */
.geo-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.geo-card-accent {
    border-left: 3px solid #58a6ff;
}
.geo-card-warn {
    border-left: 3px solid #d29922;
    background: #1c1a10;
}
.geo-card-good {
    border-left: 3px solid #3fb950;
    background: #0d1a0f;
}
.geo-card-error {
    border-left: 3px solid #f85149;
    background: #1a0d0d;
}

/* Metric chips */
.metric-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin: 0.6rem 0;
}
.metric-chip {
    background: #21262d;
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #79c0ff;
    border: 1px solid #30363d;
}
.metric-chip span {
    color: #e6edf3;
    font-weight: 600;
    margin-left: 0.4rem;
}

/* Confidence bar */
.conf-bar-container {
    background: #21262d;
    border-radius: 999px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 0.4rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #388bfd, #58a6ff);
    transition: width 0.8s ease;
}

/* Anomaly tags */
.anomaly-tag {
    display: inline-block;
    background: #2d1f00;
    border: 1px solid #d29922;
    border-radius: 4px;
    padding: 0.25rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #e3b341;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* Report section styling */
.report-section {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #c9d1d9;
}
.report-section h4 {
    font-family: 'IBM Plex Mono', monospace;
    color: #58a6ff;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 0.6rem 0;
}
.report-bullet {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.report-bullet::before {
    content: '›';
    color: #58a6ff;
    font-weight: bold;
    flex-shrink: 0;
}

/* Land cover bar */
.lc-bar { display: flex; border-radius: 4px; overflow: hidden; height: 14px; width: 100%; }
.lc-water  { background: #388bfd; }
.lc-veg    { background: #3fb950; }
.lc-urban  { background: #e3b341; }
.lc-barren { background: #8b949e; }

/* Section labels */
.label-mono {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Divider */
.geo-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.2rem 0;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #2ea043 !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #58a6ff !important;
    background: #161b22 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Pipeline imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from geospatial_platform.context import InputContext
from geospatial_platform.input_handler import handle_input
from geospatial_platform.image_processor import process_image
from geospatial_platform.vision_model import extract_vit_features
from geospatial_platform.data_integrator import integrate_data, build_climate_summary, populate_convenience_fields
from geospatial_platform.rag import retrieve_context
from geospatial_platform.llm_engine import generate_report


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_ndvi_map(ndvi_array, title="NDVI", cmap="RdYlGn"):
    """Render a spectral index map as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    vmin, vmax = (-1, 1) if "NDVI" in title or "NDWI" in title else (-1, 1)
    im = ax.imshow(ndvi_array, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="#8b949e", labelsize=7)
    cbar.outline.set_edgecolor("#21262d")
    ax.set_title(title, color="#58a6ff", fontsize=9, fontfamily="monospace", pad=6)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


def render_land_cover_bar(land_cover: dict):
    colors = {"water": "#388bfd", "vegetation": "#3fb950", "urban": "#e3b341", "barren": "#8b949e"}
    labels = {"water": "Water", "vegetation": "Vegetation", "urban": "Urban", "barren": "Barren"}
    parts = ""
    for key, pct in land_cover.items():
        if pct > 0:
            color = colors.get(key, "#555")
            parts += f'<div class="lc-{key}" style="width:{pct:.1f}%; title=\'{labels.get(key)}: {pct:.1f}%\'"></div>'
    return f'<div class="lc-bar">{parts}</div>'


def render_confidence_bar(score: float):
    color = "#3fb950" if score >= 75 else "#d29922" if score >= 55 else "#f85149"
    return f"""
    <div class="conf-bar-container">
        <div class="conf-bar-fill" style="width:{score:.0f}%; background: linear-gradient(90deg, {color}99, {color});"></div>
    </div>
    """


def parse_report_sections(report_text: str) -> dict:
    """
    Parse the markdown report into a dict of section_title -> content.
    Handles '## N. Title' format from the new LLM prompt.
    """
    import re
    sections = {}
    # Split on markdown h2 headings
    parts = re.split(r'\n##\s+', report_text)
    for part in parts:
        if not part.strip():
            continue
        lines = part.strip().split("\n", 1)
        title = lines[0].strip().lstrip("#").strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        sections[title] = content
    return sections


def render_report_section(title: str, content: str, icon: str = ""):
    """Render a single report section in a styled card."""
    st.markdown(f"""
    <div class="report-section">
        <h4>{icon} {title}</h4>
        {content.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)


# ── Section icons ─────────────────────────────────────────────────────────────
SECTION_ICONS = {
    "Executive Summary": "📋",
    "Vegetation": "🌿",
    "Temporal": "📈",
    "Hydrological": "💧",
    "Climate": "🌡️",
    "Aridity": "☀️",
    "Key Findings": "🔍",
    "Monitoring": "📡",
    "Confidence": "⚖️",
}

def get_icon(title: str) -> str:
    for key, icon in SECTION_ICONS.items():
        if key.lower() in title.lower():
            return icon
    return "📄"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; color: #58a6ff; font-weight: 600;">
            🛰️ GeoIntel
        </div>
        <div style="font-size: 0.72rem; color: #8b949e; margin-top: 0.2rem; letter-spacing: 0.05em;">
            MULTIMODAL GEOSPATIAL PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📡 Image Input")
    uploaded_t1 = st.file_uploader(
        "Primary image (or earlier date)",
        type=["tif", "tiff", "png", "jpg"],
        key="img_t1",
        help="GeoTIFF preferred. Used as the baseline or single-image analysis."
    )
    uploaded_t2 = st.file_uploader(
        "Second image — optional (later date)",
        type=["tif", "tiff", "png", "jpg"],
        key="img_t2",
        help="Upload for temporal NDVI comparison (e.g. 2024 vs 2010)."
    )

    st.markdown("### 📊 Climate Data")
    uploaded_csv = st.file_uploader(
        "NASA POWER CSV — optional",
        type=["csv"],
        key="csv",
        help="15-year monthly climate data from NASA POWER API."
    )

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Analysis Options")

    user_question = st.text_area(
        "Custom question (optional)",
        value="Provide a full scientific interpretation of this image and data.",
        height=80,
        help="Guide the report focus. Leave default for full analysis."
    )

    run_btn = st.button("▶ Run Analysis", use_container_width=True)

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.7rem; color: #484f58; line-height: 1.6;">
        Pipeline: Image → ViT → RAG → LLM<br>
        LLM: Groq / llama-3.1-8b-instant<br>
        Indices: NDVI · NDWI · NDBI
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL — HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="margin:0;">Geospatial Intelligence Platform</h1>
    <div class="label-mono" style="margin-top: 0.3rem;">
        Satellite imagery · Spectral analysis · AI-generated environmental reports
    </div>
</div>
""", unsafe_allow_html=True)

# ── Idle state ────────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div class="geo-card" style="text-align:center; padding: 3rem 2rem; border-style: dashed;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">🛰️</div>
        <div style="font-family: 'IBM Plex Mono', monospace; color: #58a6ff; font-size: 1rem; margin-bottom: 0.5rem;">
            Ready for Analysis
        </div>
        <div style="color: #8b949e; font-size: 0.85rem; max-width: 400px; margin: 0 auto;">
            Upload a satellite image in the sidebar, optionally add a second image for temporal comparison
            and a NASA POWER CSV for climate context, then click <strong>Run Analysis</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Validation ────────────────────────────────────────────────────────────────
if not uploaded_t1:
    st.markdown("""
    <div class="geo-card geo-card-error">
        ⚠️ <strong>No image uploaded.</strong> Please upload at least one satellite image in the sidebar.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

status = st.status("Running geospatial analysis pipeline…", expanded=True)

try:
    with status:
        # Step 1 — Input handling
        st.write("📥 Validating inputs and extracting metadata…")
        ic = InputContext(user_question=user_question)
        ic = handle_input(ic, uploaded_t1, uploaded_t2, uploaded_csv)

        # Step 2 — Image processing
        st.write("🔬 Computing spectral indices (NDVI · NDWI · NDBI)…")
        ic = process_image(ic)

        # Step 3 — ViT feature extraction
        st.write("🧠 Extracting Vision Transformer features…")
        ic = extract_vit_features(ic)

        # Step 4 — Climate data integration
        if uploaded_csv:
            st.write("📊 Integrating NASA POWER climate data…")
            ic = integrate_data(ic)
            ic.climate_summary = build_climate_summary(ic.climate_df)
            populate_convenience_fields(ic)
        else:
            st.write("📊 No climate CSV — skipping climate integration.")

        # Step 5 — RAG context retrieval
        st.write("📚 Retrieving environmental context (RAG)…")
        ic = retrieve_context(ic)

        # Step 6 — Report generation
        st.write("✍️ Generating scientific report…")
        ic.report = generate_report(ic, ic.rag_context or "", ic.anomalies or [])

    status.update(label="✅ Analysis complete", state="complete", expanded=False)

except Exception as e:
    status.update(label="❌ Pipeline error", state="error", expanded=True)
    st.markdown(f"""
    <div class="geo-card geo-card-error">
        <strong>Pipeline failed:</strong><br>
        <code style="font-size:0.8rem;">{e}</code>
    </div>
    """, unsafe_allow_html=True)
    st.exception(e)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_maps, tab_report, tab_raw = st.tabs([
    "  📊 Overview  ",
    "  🗺️ Index Maps  ",
    "  📄 Full Report  ",
    "  🔩 Raw Data  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:

    # ── Region & ecosystem header ─────────────────────────────────────────────
    region_str   = ic.region or "Unknown region"
    eco_str      = ic.ecosystem or "Unknown ecosystem"
    temporal_str = (
        f"{ic.temporal_label_t1} → {ic.temporal_label_t2}"
        if ic.temporal_label_t1 and ic.temporal_label_t2
        else "Single image"
    )

    st.markdown(f"""
    <div class="geo-card geo-card-accent">
        <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;">
            <div>
                <div class="label-mono">Region</div>
                <div style="font-size:1.1rem; font-weight:600; color:#e6edf3; margin-top:0.2rem;">📍 {region_str}</div>
                <div style="font-size:0.8rem; color:#8b949e; margin-top:0.2rem;">{eco_str}</div>
            </div>
            <div style="text-align:right;">
                <div class="label-mono">Temporal Coverage</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem; color:#79c0ff; margin-top:0.2rem;">
                    📅 {temporal_str}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    # ── Spectral metrics ──────────────────────────────────────────────────────
    with col1:
        st.markdown("## Spectral Indices")

        def chip(label, value, unit=""):
            return f'<div class="metric-chip">{label}<span>{value:.3f}{unit}</span></div>'

        chips = ""
        if ic.ndvi_mean is not None:
            chips += chip("NDVI", ic.ndvi_mean)
        if ic.ndwi_mean is not None:
            chips += chip("NDWI", ic.ndwi_mean)
        if ic.ndbi_mean is not None:
            chips += chip("NDBI", ic.ndbi_mean)
        if ic.aridity_index is not None:
            chips += chip("Aridity", ic.aridity_index)
        if ic.ndvi_delta is not None:
            sign = "+" if ic.ndvi_delta >= 0 else ""
            chips += f'<div class="metric-chip">ΔNDVI<span style="color:{"#3fb950" if ic.ndvi_delta>=0 else "#f85149"}">{sign}{ic.ndvi_delta:.3f}</span></div>'

        st.markdown(f'<div class="metric-row">{chips}</div>', unsafe_allow_html=True)

        # ── Land cover ────────────────────────────────────────────────────────
        if ic.land_cover:
            st.markdown("## Land Cover")
            st.markdown(render_land_cover_bar(ic.land_cover), unsafe_allow_html=True)

            lc_cols = st.columns(4)
            lc_colors = {"water":"#388bfd", "vegetation":"#3fb950", "urban":"#e3b341", "barren":"#8b949e"}
            for i, (cls, pct) in enumerate(ic.land_cover.items()):
                with lc_cols[i % 4]:
                    color = lc_colors.get(cls, "#8b949e")
                    st.markdown(f"""
                    <div style="text-align:center; margin-top:0.5rem;">
                        <div style="width:12px;height:12px;background:{color};border-radius:2px;margin:0 auto 3px;"></div>
                        <div class="label-mono">{cls}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;color:#e6edf3;font-weight:600;">
                            {pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Anomalies ─────────────────────────────────────────────────────────
        if ic.anomalies:
            st.markdown("## Detected Anomalies")
            tags = "".join(f'<span class="anomaly-tag">⚠ {a}</span>' for a in ic.anomalies)
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)

    # ── Confidence + climate summary ──────────────────────────────────────────
    with col2:
        st.markdown("## Confidence Score")
        score = ic.confidence_score or 0
        label_color = "#3fb950" if score >= 75 else "#d29922" if score >= 55 else "#f85149"
        st.markdown(f"""
        <div class="geo-card">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;color:{label_color};font-weight:600;">
                    {score:.0f}%
                </div>
                <div class="label-mono">{'High' if score>=75 else 'Moderate' if score>=55 else 'Low'} confidence</div>
            </div>
            {render_confidence_bar(score)}
        </div>
        """, unsafe_allow_html=True)

        # ── Image metadata ────────────────────────────────────────────────────
        st.markdown("## Image Metadata")
        meta_items = []
        if ic.image_format:   meta_items.append(("Format", ic.image_format))
        if ic.image_bands:    meta_items.append(("Bands", str(ic.image_bands)))
        if ic.image_dims:     meta_items.append(("Dimensions", f"{ic.image_dims[0]} × {ic.image_dims[1]} px"))
        if ic.image_crs:      meta_items.append(("CRS", ic.image_crs))

        rows = "".join(f"""
        <div style="display:flex;justify-content:space-between;padding:0.35rem 0;
                    border-bottom:1px solid #21262d;font-size:0.82rem;">
            <span class="label-mono">{k}</span>
            <span style="font-family:'IBM Plex Mono',monospace;color:#c9d1d9;">{v}</span>
        </div>
        """ for k, v in meta_items)

        st.markdown(f'<div class="geo-card">{rows}</div>', unsafe_allow_html=True)

        # ── Climate snapshot ──────────────────────────────────────────────────
        if ic.climate_summary:
            cs = ic.climate_summary
            st.markdown("## Climate Snapshot")
            climate_rows = ""
            pairs = [
                ("Rainfall (latest)", f"{cs.get('rainfall_mm_latest',0):.1f} mm"),
                ("Rainfall trend",    cs.get('rainfall_mm_trend','—').capitalize()),
                ("Temperature",       f"{cs.get('temperature_c_latest',0):.1f} °C"),
                ("Humidity",          f"{cs.get('humidity_pct_latest',0):.1f} %"),
            ]
            for k, v in pairs:
                climate_rows += f"""
                <div style="display:flex;justify-content:space-between;padding:0.35rem 0;
                            border-bottom:1px solid #21262d;font-size:0.82rem;">
                    <span class="label-mono">{k}</span>
                    <span style="font-family:'IBM Plex Mono',monospace;color:#c9d1d9;">{v}</span>
                </div>
                """
            st.markdown(f'<div class="geo-card">{climate_rows}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="geo-card" style="text-align:center;color:#484f58;font-size:0.8rem;padding:1.5rem;">
                No climate CSV uploaded.<br>Upload NASA POWER data for climate analysis.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INDEX MAPS
# ══════════════════════════════════════════════════════════════════════════════

with tab_maps:
    st.markdown("## Spectral Index Maps")

    maps_available = any([
        ic.ndvi_map is not None,
        ic.ndwi_map is not None,
        ic.ndbi_map is not None,
    ])

    if not maps_available:
        st.markdown("""
        <div class="geo-card" style="text-align:center;color:#484f58;padding:2rem;">
            No index maps available — check that image processing completed successfully.
        </div>
        """, unsafe_allow_html=True)
    else:
        map_configs = [
            (ic.ndvi_map, "NDVI — Vegetation Density", "RdYlGn"),
            (ic.ndwi_map, "NDWI — Water Content",      "Blues"),
            (ic.ndbi_map, "NDBI — Built-up Index",     "YlOrRd"),
        ]
        available_maps = [(arr, title, cmap) for arr, title, cmap in map_configs if arr is not None]
        cols = st.columns(len(available_maps))

        for col, (arr, title, cmap) in zip(cols, available_maps):
            with col:
                fig = render_ndvi_map(arr, title, cmap)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # ── Temporal NDVI comparison ──────────────────────────────────────────────
    if ic.ndvi_map is not None and ic.ndvi_mean_t1 is not None and ic.ndvi_mean_t2 is not None:
        st.markdown("## Temporal NDVI Comparison")
        st.markdown(f"""
        <div class="geo-card geo-card-{'good' if ic.ndvi_delta >= 0 else 'error'}">
            <div style="display:flex; gap:2rem; align-items:center; flex-wrap:wrap;">
                <div>
                    <div class="label-mono">{ic.temporal_label_t1 or 'Image 1'}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;color:#e6edf3;">
                        {ic.ndvi_mean_t1:.3f}
                    </div>
                </div>
                <div style="font-size:1.5rem;color:#8b949e;">→</div>
                <div>
                    <div class="label-mono">{ic.temporal_label_t2 or 'Image 2'}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;color:#e6edf3;">
                        {ic.ndvi_mean_t2:.3f}
                    </div>
                </div>
                <div style="margin-left:1rem;">
                    <div class="label-mono">Change (ΔNDVI)</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;
                                color:{'#3fb950' if ic.ndvi_delta>=0 else '#f85149'};font-weight:600;">
                        {'↑ +' if ic.ndvi_delta>=0 else '↓ '}{ic.ndvi_delta:.3f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FULL REPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_report:
    if not ic.report:
        st.markdown("""
        <div class="geo-card" style="text-align:center;color:#484f58;padding:2rem;">
            Report not generated — check pipeline logs.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Strip the ====== header/footer wrapper before parsing
        import re
        report_body = re.sub(r'^={20,}.*?={20,}\n', '', ic.report, flags=re.DOTALL).strip()
        report_body = re.sub(r'={20,}.*$', '', report_body, flags=re.DOTALL).strip()

        sections = parse_report_sections(report_body)

        if sections:
            for title, content in sections.items():
                icon = get_icon(title)
                render_report_section(title, content, icon)
        else:
            # Fallback: render raw report with basic formatting
            st.markdown(f"""
            <div class="report-section" style="white-space:pre-wrap;">
                {ic.report}
            </div>
            """, unsafe_allow_html=True)

        # ── Download button ───────────────────────────────────────────────────
        st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="⬇ Download Report (.txt)",
                data=ic.report,
                file_name=f"geointel_report_{ic.region or 'unknown'}.txt".replace(" ", "_"),
                mime="text/plain",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

with tab_raw:
    st.markdown("## Raw Pipeline Data")

    with st.expander("📚 Retrieved RAG Context", expanded=False):
        if ic.rag_context:
            st.code(ic.rag_context, language=None)
        else:
            st.caption("No RAG context retrieved.")

    with st.expander("🌡️ Climate Summary Dict", expanded=False):
        if ic.climate_summary:
            st.json(ic.climate_summary)
        else:
            st.caption("No climate data loaded.")

    with st.expander("📋 Full InputContext Fields", expanded=False):
        from dataclasses import asdict
        try:
            ctx_dict = {
                k: (v.tolist() if hasattr(v, 'tolist') else str(v) if not isinstance(v, (str, int, float, bool, type(None), dict, list)) else v)
                for k, v in asdict(ic).items()
                if k not in ("ndvi_map", "ndwi_map", "ndbi_map", "vit_features", "climate_df")
            }
            st.json(ctx_dict)
        except Exception as e:
            st.caption(f"Could not serialise context: {e}")

    with st.expander("📝 Raw LLM Report Text", expanded=False):
        if ic.report:
            st.code(ic.report, language=None)
        else:
            st.caption("No report generated.")
