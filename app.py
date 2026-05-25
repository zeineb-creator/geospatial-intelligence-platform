import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io, sys, os, tempfile, re
from geospatial_platform.vision_model import extract_vit_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geospatial Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — light theme ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Background */
.stApp { background-color: #f5f7fa; color: #1a1f2e; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #dde3ed;
}
section[data-testid="stSidebar"] * { color: #1a1f2e !important; }

/* Headers */
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.02em;
}
h1 { font-size: 1.6rem !important; color: #1a1f2e !important; }
h2 { font-size: 1.05rem !important; color: #2563eb !important;
     border-bottom: 1px solid #dde3ed; padding-bottom: 0.4rem; }
h3 { font-size: 0.95rem !important; color: #3b82f6 !important; }

/* Cards */
.geo-card {
    background: #ffffff;
    border: 1px solid #dde3ed;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.geo-card-accent { border-left: 3px solid #2563eb; }
.geo-card-warn   { border-left: 3px solid #d97706; background: #fffbeb; }
.geo-card-good   { border-left: 3px solid #16a34a; background: #f0fdf4; }
.geo-card-error  { border-left: 3px solid #dc2626; background: #fef2f2; }

/* Metric chips */
.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0.6rem 0; }
.metric-chip {
    background: #f0f4ff;
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #2563eb;
    border: 1px solid #c7d7fc;
}
.metric-chip span {
    color: #1a1f2e;
    font-weight: 600;
    margin-left: 0.4rem;
}

/* Confidence bar */
.conf-bar-container {
    background: #e5e7eb;
    border-radius: 999px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 0.4rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}

/* Anomaly tags */
.anomaly-tag {
    display: inline-block;
    background: #fffbeb;
    border: 1px solid #d97706;
    border-radius: 4px;
    padding: 0.25rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #92400e;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* Report sections */
.report-section {
    background: #ffffff;
    border: 1px solid #dde3ed;
    border-radius: 6px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
    font-size: 0.88rem;
    line-height: 1.75;
    color: #374151;
}
.report-section h4 {
    font-family: 'IBM Plex Mono', monospace;
    color: #2563eb;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 0.6rem 0;
}

/* Land cover bar */
.lc-bar { display: flex; border-radius: 4px; overflow: hidden; height: 14px; width: 100%; }
.lc-water  { background: #3b82f6; }
.lc-veg    { background: #22c55e; }
.lc-urban  { background: #f59e0b; }
.lc-barren { background: #94a3b8; }

/* Section labels */
.label-mono {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Divider */
.geo-divider { border: none; border-top: 1px solid #dde3ed; margin: 1.2rem 0; }

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #f8fafc !important;
    border: 1px dashed #c7d7fc !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.2s;
}
.stButton > button:hover { background: #1d4ed8 !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #6b7280;
}
.stTabs [aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #2563eb !important;
    background: #f8fafc !important;
}

/* Table rows */
.meta-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid #f0f4ff;
    font-size: 0.82rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))


# ── Utility helpers ───────────────────────────────────────────────────────────

def save_upload_to_temp(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp file and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def render_index_map(arr, title, cmap):
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    im = ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="#6b7280", labelsize=7)
    cbar.outline.set_edgecolor("#dde3ed")
    ax.set_title(title, color="#2563eb", fontsize=9, fontfamily="monospace", pad=6)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


def render_land_cover_bar(land_cover: dict) -> str:
    parts = ""
    for key, pct in land_cover.items():
        if pct > 0:
            parts += f'<div class="lc-{key}" style="width:{pct:.1f}%"></div>'
    return f'<div class="lc-bar">{parts}</div>'


def render_confidence_bar(score: float) -> str:
    color = "#16a34a" if score >= 75 else "#d97706" if score >= 55 else "#dc2626"
    return f"""
    <div class="conf-bar-container">
        <div class="conf-bar-fill" style="width:{score:.0f}%;background:{color};"></div>
    </div>"""


def parse_report_sections(report_text: str) -> dict:
    sections = {}
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
    st.markdown(f"""
    <div class="report-section">
        <h4>{icon} {title}</h4>
        {content.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)


SECTION_ICONS = {
    "Executive": "📋", "Vegetation": "🌿", "Temporal": "📈",
    "Hydrological": "💧", "Climate": "🌡️", "Aridity": "☀️",
    "Key Findings": "🔍", "Monitoring": "📡", "Confidence": "⚖️",
}
def get_icon(title):
    for key, icon in SECTION_ICONS.items():
        if key.lower() in title.lower():
            return icon
    return "📄"


def import_error_card(module: str, err: Exception):
    st.markdown(f"""
    <div class="geo-card geo-card-error">
        <strong>Import failed:</strong> <code>{module}</code><br><br>
        <code style="font-size:0.78rem;">{err}</code><br><br>
        <span style="color:#6b7280;font-size:0.78rem;">
        Check that the function name in <code>{module}</code> matches what app.py expects.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.2rem 0;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;
                    color:#2563eb;font-weight:600;">🛰️ GeoIntel</div>
        <div style="font-size:0.70rem;color:#6b7280;margin-top:0.2rem;letter-spacing:0.05em;">
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
        help="Upload for temporal NDVI comparison."
    )

    st.markdown("### 📊 Climate Data")
    uploaded_csv = st.file_uploader(
        "NASA POWER CSV — optional",
        type=["csv"],
        key="csv",
        help="15-year monthly climate data from NASA POWER API."
    )

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Options")
    user_question = st.text_area(
        "Custom question (optional)",
        value="Provide a full scientific interpretation of this image and data.",
        height=80,
    )
    run_btn = st.button("▶ Run Analysis", use_container_width=True)

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#9ca3af;line-height:1.7;">
        Pipeline: Image → ViT → RAG → LLM<br>
        LLM: Groq / llama-3.1-8b-instant<br>
        Indices: NDVI · NDWI · NDBI
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:1.5rem;">
    <h1 style="margin:0;">Geospatial Intelligence Platform</h1>
    <div class="label-mono" style="margin-top:0.3rem;">
        Satellite imagery · Spectral analysis · AI-generated environmental reports
    </div>
</div>
""", unsafe_allow_html=True)

# ── Idle state ────────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div class="geo-card" style="text-align:center;padding:3rem 2rem;border-style:dashed;">
        <div style="font-size:2.5rem;margin-bottom:1rem;">🛰️</div>
        <div style="font-family:'IBM Plex Mono',monospace;color:#2563eb;font-size:1rem;margin-bottom:0.5rem;">
            Ready for Analysis
        </div>
        <div style="color:#6b7280;font-size:0.85rem;max-width:420px;margin:0 auto;">
            Upload a satellite image in the sidebar, optionally add a second image
            for temporal comparison and a NASA POWER CSV, then click
            <strong>Run Analysis</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not uploaded_t1:
    st.markdown("""
    <div class="geo-card geo-card-error">
        ⚠️ <strong>No image uploaded.</strong>
        Please upload at least one satellite image in the sidebar.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

status = st.status("Running geospatial analysis pipeline…", expanded=True)
temp_files = []   # track for cleanup

try:
    with status:

        # ── Lazy imports ──────────────────────────────────────────────────────
        st.write("📦 Loading pipeline modules…")

        try:
            from geospatial_platform.context import InputContext
        except Exception as e:
            import_error_card("geospatial_platform.context → InputContext", e); st.stop()

        try:
            from geospatial_platform.input_handler import build_input_context
        except Exception as e:
            import_error_card("geospatial_platform.input_handler → build_input_context", e); st.stop()

        try:
            from geospatial_platform.image_processor import process_image
        except Exception as e:
            import_error_card("geospatial_platform.image_processor → process_image", e); st.stop()

        try:
            from geospatial_platform.vision_model import extract_vit_features
        except Exception as e:
            import_error_card("geospatial_platform.vision_model → extract_vit_features", e); st.stop()

        build_climate_summary = None
        populate_convenience_fields = None
        try:
            from geospatial_platform.data_integrator import (
                integrate_data, build_climate_summary, populate_convenience_fields,
            )
        except ImportError:
            try:
                from geospatial_platform.data_integrator import integrate_data
                st.warning(
                    "⚠️ build_climate_summary / populate_convenience_fields not yet added "
                    "to data_integrator.py. Climate summary will be skipped."
                )
            except Exception as e:
                import_error_card("geospatial_platform.data_integrator → integrate_data", e); st.stop()

        try:
            from geospatial_platform.rag import retrieve_context
        except Exception as e:
            import_error_card("geospatial_platform.rag → retrieve_context", e); st.stop()

        try:
            from geospatial_platform.llm_engine import generate_report
        except Exception as e:
            import_error_card("geospatial_platform.llm_engine → generate_report", e); st.stop()

        # ── Save uploads to temp files ────────────────────────────────────────
        st.write("📥 Saving uploaded files…")
        path_t1 = save_upload_to_temp(uploaded_t1)
        temp_files.append(path_t1)

        path_t2 = None
        if uploaded_t2:
            path_t2 = save_upload_to_temp(uploaded_t2)
            temp_files.append(path_t2)

        path_csv = None
        if uploaded_csv:
            path_csv = save_upload_to_temp(uploaded_csv)
            temp_files.append(path_csv)

        # ── Step 1 — Input handling ───────────────────────────────────────────
        st.write("📥 Validating inputs and extracting metadata…")
        ic = build_input_context(
            image_path=path_t1,
            csv_path=path_csv,
            question=user_question,
        )

        # ── Step 1b — Second image (temporal) ────────────────────────────────
        if path_t2:
            st.write("📅 Loading second image for temporal comparison…")
            from geospatial_platform.input_handler import load_image
            array_t2, meta_t2, _, _ = load_image(path_t2)
            ic.image_array_t2 = array_t2
            ic.image_meta_t2  = meta_t2

        # ── Step 2 — Image processing ─────────────────────────────────────────
        st.write("🔬 Computing spectral indices (NDVI · NDWI · NDBI)…")
        ic = process_image(ic)

        # ── Step 3 — ViT feature extraction ──────────────────────────────────
        st.write("🧠 Extracting Vision Transformer features…")
        ic = extract_vit_features(ic)

        # ── Step 4 — Climate data integration ─────────────────────────────────
        if path_csv:
            st.write("📊 Integrating NASA POWER climate data…")
            ic = integrate_data(ic)
            if build_climate_summary and populate_convenience_fields:
                ic.climate_summary = build_climate_summary(ic.climate_df)
                populate_convenience_fields(ic)
        else:
            st.write("📊 No climate CSV — skipping climate integration.")

        # ── Step 5 — RAG ──────────────────────────────────────────────────────
        st.write("📚 Retrieving environmental context (RAG)…")
        ic = retrieve_context(ic)

        # ── Step 6 — Report generation ────────────────────────────────────────
        st.write("✍️ Generating scientific report…")
        ic.report = generate_report(ic, ic.rag_context or "", ic.anomalies or [])

    status.update(label="✅ Analysis complete", state="complete", expanded=False)

except Exception as e:
    status.update(label="❌ Pipeline error", state="error", expanded=True)
    st.markdown(f"""
    <div class="geo-card geo-card-error">
        <strong>Pipeline failed at runtime:</strong><br><br>
        <code style="font-size:0.78rem;">{type(e).__name__}: {e}</code>
    </div>
    """, unsafe_allow_html=True)
    st.exception(e)
    st.stop()

finally:
    # Clean up temp files
    for p in temp_files:
        try: os.unlink(p)
        except: pass


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

    # Region header
    region_str = getattr(ic, 'region', None) or ic.image_meta.get("region_name", "Unknown region")
    eco_str    = getattr(ic, 'ecosystem', None) or "—"
    t1_label   = getattr(ic, 'temporal_label_t1', None)
    t2_label   = getattr(ic, 'temporal_label_t2', None)
    temporal_str = f"{t1_label} → {t2_label}" if t1_label and t2_label else "Single image"

    st.markdown(f"""
    <div class="geo-card geo-card-accent">
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
            <div>
                <div class="label-mono">Region</div>
                <div style="font-size:1.05rem;font-weight:600;color:#1a1f2e;margin-top:0.2rem;">
                    📍 {region_str}
                </div>
                <div style="font-size:0.8rem;color:#6b7280;margin-top:0.2rem;">{eco_str}</div>
            </div>
            <div style="text-align:right;">
                <div class="label-mono">Temporal Coverage</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                            color:#2563eb;margin-top:0.2rem;">📅 {temporal_str}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Spectral index chips
        st.markdown("## Spectral Indices")
        ndvi = getattr(ic, 'ndvi_mean', None)
        ndwi = getattr(ic, 'ndwi_mean', None)
        ndbi = getattr(ic, 'ndbi_mean', None)
        ai   = getattr(ic, 'aridity_index', None)
        delta = getattr(ic, 'ndvi_delta', None)

        chips = ""
        if ndvi  is not None: chips += f'<div class="metric-chip">NDVI<span>{ndvi:.3f}</span></div>'
        if ndwi  is not None: chips += f'<div class="metric-chip">NDWI<span>{ndwi:.3f}</span></div>'
        if ndbi  is not None: chips += f'<div class="metric-chip">NDBI<span>{ndbi:.3f}</span></div>'
        if ai    is not None: chips += f'<div class="metric-chip">Aridity<span>{ai:.3f}</span></div>'
        if delta is not None:
            sign  = "+" if delta >= 0 else ""
            color = "#16a34a" if delta >= 0 else "#dc2626"
            chips += f'<div class="metric-chip">ΔNDVI<span style="color:{color}">{sign}{delta:.3f}</span></div>'
        st.markdown(f'<div class="metric-row">{chips}</div>', unsafe_allow_html=True)

        # Land cover
        land_cover = getattr(ic, 'land_cover', None)
        if land_cover:
            st.markdown("## Land Cover")
            st.markdown(render_land_cover_bar(land_cover), unsafe_allow_html=True)
            lc_cols = st.columns(4)
            lc_colors = {"water":"#3b82f6","vegetation":"#22c55e","urban":"#f59e0b","barren":"#94a3b8"}
            for i, (cls, pct) in enumerate(land_cover.items()):
                with lc_cols[i % 4]:
                    color = lc_colors.get(cls, "#94a3b8")
                    st.markdown(f"""
                    <div style="text-align:center;margin-top:0.5rem;">
                        <div style="width:12px;height:12px;background:{color};border-radius:2px;margin:0 auto 3px;"></div>
                        <div class="label-mono">{cls}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;
                                    color:#1a1f2e;font-weight:600;">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Anomalies
        anomalies = getattr(ic, 'anomalies', None)
        if anomalies:
            st.markdown("## Detected Anomalies")
            tags = "".join(f'<span class="anomaly-tag">⚠ {a}</span>' for a in anomalies)
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)

    with col2:
        # Confidence score
        st.markdown("## Confidence Score")
        score = getattr(ic, 'confidence_score', None) or 0
        label_color = "#16a34a" if score >= 75 else "#d97706" if score >= 55 else "#dc2626"
        label_text  = "High" if score >= 75 else "Moderate" if score >= 55 else "Low"
        st.markdown(f"""
        <div class="geo-card">
            <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;
                            color:{label_color};font-weight:600;">{score:.0f}%</div>
                <div class="label-mono">{label_text} confidence</div>
            </div>
            {render_confidence_bar(score)}
        </div>
        """, unsafe_allow_html=True)

        # Image metadata
        st.markdown("## Image Metadata")
        meta = ic.image_meta
        meta_items = [
            ("Format",     getattr(ic, 'image_format', '—')),
            ("Bands",      str(getattr(ic, 'n_bands', '—'))),
            ("Dimensions", f"{meta.get('width','?')} × {meta.get('height','?')} px"),
            ("CRS",        meta.get('crs', '—')),
            ("Context",    meta.get('region_context', '—')),
        ]
        rows = "".join(f"""
        <div class="meta-row">
            <span class="label-mono">{k}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                         color:#374151;text-align:right;max-width:55%;">{v}</span>
        </div>""" for k, v in meta_items)
        st.markdown(f'<div class="geo-card">{rows}</div>', unsafe_allow_html=True)

        # Climate snapshot
        climate_summary = getattr(ic, 'climate_summary', None)
        if climate_summary:
            st.markdown("## Climate Snapshot")
            cs = climate_summary
            pairs = [
                ("Rainfall (latest)", f"{cs.get('rainfall_mm_latest',0):.1f} mm"),
                ("Rainfall trend",    cs.get('rainfall_mm_trend','—').capitalize()),
                ("Temperature",       f"{cs.get('temperature_c_latest',0):.1f} °C"),
                ("Humidity",          f"{cs.get('humidity_pct_latest',0):.1f} %"),
            ]
            rows = "".join(f"""
            <div class="meta-row">
                <span class="label-mono">{k}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                             color:#374151;">{v}</span>
            </div>""" for k, v in pairs)
            st.markdown(f'<div class="geo-card">{rows}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="geo-card" style="text-align:center;color:#9ca3af;
                         font-size:0.8rem;padding:1.5rem;">
                No climate CSV uploaded.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INDEX MAPS
# ══════════════════════════════════════════════════════════════════════════════

with tab_maps:
    st.markdown("## Spectral Index Maps")

    ndvi_map = getattr(ic, 'ndvi_map', None)
    ndwi_map = getattr(ic, 'ndwi_map', None)
    ndbi_map = getattr(ic, 'ndbi_map', None)

    map_configs = [
        (ndvi_map, "NDVI — Vegetation Density", "RdYlGn"),
        (ndwi_map, "NDWI — Water Content",      "Blues_r"),
        (ndbi_map, "NDBI — Built-up Index",     "YlOrRd"),
    ]
    available = [(a, t, c) for a, t, c in map_configs if a is not None]

    if not available:
        st.markdown("""
        <div class="geo-card" style="text-align:center;color:#9ca3af;padding:2rem;">
            No index maps available — image processing may have returned None arrays.
        </div>
        """, unsafe_allow_html=True)
    else:
        cols = st.columns(len(available))
        for col, (arr, title, cmap) in zip(cols, available):
            with col:
                fig = render_index_map(arr, title, cmap)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # Temporal NDVI comparison
    ndvi_delta = getattr(ic, 'ndvi_delta', None)
    ndvi_t1    = getattr(ic, 'ndvi_mean_t1', None)
    ndvi_t2    = getattr(ic, 'ndvi_mean_t2', None)

    if ndvi_delta is not None and ndvi_t1 is not None and ndvi_t2 is not None:
        st.markdown("## Temporal NDVI Comparison")
        card_class = "geo-card-good" if ndvi_delta >= 0 else "geo-card-error"
        arrow = f"↑ +{ndvi_delta:.3f}" if ndvi_delta >= 0 else f"↓ {ndvi_delta:.3f}"
        arrow_color = "#16a34a" if ndvi_delta >= 0 else "#dc2626"
        st.markdown(f"""
        <div class="geo-card {card_class}">
            <div style="display:flex;gap:2.5rem;align-items:center;flex-wrap:wrap;">
                <div>
                    <div class="label-mono">{t1_label or 'Image 1'}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                                color:#1a1f2e;">{ndvi_t1:.3f}</div>
                </div>
                <div style="font-size:1.4rem;color:#9ca3af;">→</div>
                <div>
                    <div class="label-mono">{t2_label or 'Image 2'}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                                color:#1a1f2e;">{ndvi_t2:.3f}</div>
                </div>
                <div style="margin-left:1rem;">
                    <div class="label-mono">Change (ΔNDVI)</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;
                                color:{arrow_color};font-weight:700;">{arrow}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FULL REPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_report:
    report = getattr(ic, 'report', None)
    if not report:
        st.markdown("""
        <div class="geo-card" style="text-align:center;color:#9ca3af;padding:2rem;">
            Report not generated.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Strip === header/footer wrappers
        report_body = re.sub(r'^={20,}.*?={20,}\n', '', report, flags=re.DOTALL).strip()
        report_body = re.sub(r'\n={20,}.*$', '', report_body, flags=re.DOTALL).strip()

        sections = parse_report_sections(report_body)
        if sections:
            for title, content in sections.items():
                render_report_section(title, content, get_icon(title))
        else:
            st.markdown(f"""
            <div class="report-section" style="white-space:pre-wrap;">{report}</div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="⬇ Download Report (.txt)",
                data=report,
                file_name=f"geointel_{region_str.replace(' ','_').replace(',','')}.txt",
                mime="text/plain",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

with tab_raw:
    st.markdown("## Raw Pipeline Data")

    with st.expander("📚 Retrieved RAG Context"):
        rag = getattr(ic, 'rag_context', None)
        st.code(rag or "None", language=None)

    with st.expander("🌡️ Climate Summary"):
        cs = getattr(ic, 'climate_summary', None)
        if cs: st.json(cs)
        else:  st.caption("No climate data.")

    with st.expander("📋 InputContext Fields"):
        safe = {}
        for k, v in vars(ic).items():
            if k in ("ndvi_map", "ndwi_map", "ndbi_map", "vit_features",
                     "image_array", "image_array_t2", "climate_df", "csv_df"):
                safe[k] = f"<{type(v).__name__} — omitted>"
            else:
                safe[k] = str(v) if not isinstance(v, (str, int, float, bool, type(None), dict, list)) else v
        st.json(safe)

    with st.expander("📝 Raw LLM Report"):
        st.code(getattr(ic, 'report', '') or "None", language=None)
