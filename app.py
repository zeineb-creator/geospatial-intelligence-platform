"""
app.py — Multimodal Geospatial Intelligence Platform
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, sys, os, tempfile, re
from datetime import date as _date

st.set_page_config(
    page_title="Geospatial Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.stApp{background-color:#f5f7fa;color:#1a1f2e;}
section[data-testid="stSidebar"]{background-color:#ffffff;border-right:1px solid #dde3ed;}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div.stMarkdown{color:#1a1f2e !important;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace !important;letter-spacing:-0.02em;}
h1{font-size:1.6rem !important;color:#1a1f2e !important;}
h2{font-size:1.05rem !important;color:#2563eb !important;border-bottom:1px solid #dde3ed;padding-bottom:0.4rem;}
h3{font-size:0.95rem !important;color:#3b82f6 !important;}
.geo-card{background:#ffffff;border:1px solid #dde3ed;border-radius:8px;padding:1.1rem 1.3rem;margin-bottom:1rem;}
.geo-card-accent{border-left:3px solid #2563eb;}
.geo-card-warn{border-left:3px solid #d97706;background:#fffbeb;}
.geo-card-good{border-left:3px solid #16a34a;background:#f0fdf4;}
.geo-card-error{border-left:3px solid #dc2626;background:#fef2f2;}
.geo-card-guide{border-left:3px solid #7c3aed;background:#f5f3ff;}
.metric-row{display:flex;gap:0.8rem;flex-wrap:wrap;margin:0.6rem 0;}
.metric-chip{background:#f0f4ff;border-radius:6px;padding:0.5rem 0.9rem;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#2563eb;border:1px solid #c7d7fc;}
.metric-chip span{color:#1a1f2e;font-weight:600;margin-left:0.4rem;}
.conf-bar-container{background:#e5e7eb;border-radius:999px;height:8px;width:100%;overflow:hidden;margin-top:0.4rem;}
.conf-bar-fill{height:100%;border-radius:999px;transition:width 0.8s ease;}
.anomaly-tag{display:inline-block;background:#fffbeb;border:1px solid #d97706;border-radius:4px;padding:0.25rem 0.6rem;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#92400e;margin:0.2rem 0.2rem 0.2rem 0;}
.report-section{background:#ffffff;border:1px solid #dde3ed;border-radius:6px;padding:1rem 1.3rem;margin-bottom:0.8rem;font-size:0.88rem;line-height:1.75;color:#374151;}
.report-section h4{font-family:'IBM Plex Mono',monospace;color:#2563eb;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.6rem 0;}
.guide-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:0.75rem;margin-top:0.8rem;}
.guide-chip{background:#ffffff;border:1px solid #dde3ed;border-radius:8px;padding:0.8rem 1rem;font-size:0.82rem;line-height:1.6;color:#374151;}
.guide-chip strong{display:block;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#7c3aed;margin-bottom:0.25rem;letter-spacing:0.04em;}
.guide-scale{display:flex;gap:0.3rem;margin-top:0.5rem;font-size:0.7rem;font-family:'IBM Plex Mono',monospace;}
.guide-scale-block{flex:1;height:6px;border-radius:3px;}
.lc-bar{display:flex;border-radius:4px;overflow:hidden;height:14px;width:100%;}
.lc-water{background:#3b82f6;}.lc-veg{background:#22c55e;}.lc-urban{background:#f59e0b;}.lc-barren{background:#94a3b8;}
.label-mono{font-family:'IBM Plex Mono',monospace;font-size:0.70rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;}
.geo-divider{border:none;border-top:1px solid #dde3ed;margin:1.2rem 0;}
[data-testid="stFileUploader"]{background:#f8fafc !important;border:1px dashed #c7d7fc !important;border-radius:8px !important;}
.stButton>button{background:#2563eb !important;color:#fff !important;border:none !important;border-radius:6px !important;font-family:'IBM Plex Mono',monospace !important;font-size:0.85rem !important;padding:0.5rem 1.5rem !important;}
.stButton>button:hover{background:#1d4ed8 !important;}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#6b7280;}
.stTabs [aria-selected="true"]{color:#2563eb !important;border-bottom-color:#2563eb !important;}
.streamlit-expanderHeader{font-family:'IBM Plex Mono',monospace !important;font-size:0.82rem !important;color:#2563eb !important;background:#f8fafc !important;}
.meta-row{display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid #f0f4ff;font-size:0.82rem;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
[data-testid="collapsedControl"]{visibility:visible !important;display:flex !important;}
button[kind="header"]{visibility:visible !important;}
[data-testid="stSidebarCollapseButton"]{visibility:visible !important;}
</style>
""", unsafe_allow_html=True)

sys.path.insert(0, os.path.dirname(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def save_upload_to_temp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def _extract_lat_from_meta(meta):
    try:
        import rasterio.transform as rt
        t = meta.get("transform"); h = meta.get("height",100); w = meta.get("width",100)
        if t:
            _, y = rt.xy(t, h//2, w//2); return float(y)
    except Exception: pass
    return meta.get("lat", None)

def _extract_lon_from_meta(meta):
    try:
        import rasterio.transform as rt
        t = meta.get("transform"); h = meta.get("height",100); w = meta.get("width",100)
        if t:
            x, _ = rt.xy(t, h//2, w//2); return float(x)
    except Exception: pass
    return meta.get("lon", None)

def render_index_map(arr, title, cmap):
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    im = ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="#6b7280", labelsize=7)
    cbar.outline.set_edgecolor("#dde3ed")
    ax.set_title(title, color="#2563eb", fontsize=9, fontfamily="monospace", pad=6)
    ax.axis("off"); fig.tight_layout(pad=0.5)
    return fig

def render_land_cover_bar(land_cover):
    parts = ""
    for key, pct in land_cover.items():
        if pct > 0:
            parts += '<div class="lc-' + key + '" style="width:' + str(round(pct,1)) + '%"></div>'
    return '<div class="lc-bar">' + parts + '</div>'

def render_confidence_bar(score):
    color = "#16a34a" if score >= 75 else "#d97706" if score >= 55 else "#dc2626"
    return ('<div class="conf-bar-container">'
            '<div class="conf-bar-fill" style="width:' + str(round(score)) + '%;background:' + color + ';"></div>'
            '</div>')

def parse_report_sections(report_text):
    sections = {}
    for part in re.split(r'\n##\s+', report_text):
        if not part.strip(): continue
        lines = part.strip().split("\n", 1)
        title   = lines[0].strip().lstrip("#").strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        sections[title] = content
    return sections

def render_report_section(title, content, icon=""):
    safe_content = content.replace(chr(10), '<br>')
    st.markdown(
        '<div class="report-section"><h4>' + icon + ' ' + title + '</h4>' + safe_content + '</div>',
        unsafe_allow_html=True
    )

SECTION_ICONS = {
    "Executive":"📋","Vegetation":"🌿","Temporal":"📈",
    "Hydrological":"💧","Climate":"🌡️","Aridity":"☀️",
    "Key Findings":"🔍","Monitoring":"📡","Confidence":"⚖️",
}
def get_icon(title):
    for key, icon in SECTION_ICONS.items():
        if key.lower() in title.lower(): return icon
    return "📄"

def import_error_card(module, err):
    st.markdown(
        '<div class="geo-card geo-card-error"><strong>Import failed:</strong> <code>' +
        module + '</code><br><br><code style="font-size:0.78rem;">' + str(err) + '</code></div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# USER GUIDE
# ══════════════════════════════════════════════════════════════════════════════

def render_user_guide():
    st.markdown(
        '<div class="geo-card geo-card-guide" style="margin-bottom:1rem;">'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;color:#7c3aed;'
        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">📖 How to read this platform — no expertise required</div>'
        '<div style="font-size:0.84rem;color:#374151;line-height:1.7;">'
        'This platform analyses satellite images and computes <strong>spectral indices</strong> — '
        'numbers that reveal what the land looks like from space. Think of them as colour filters '
        'highlighting vegetation, water, buildings, or bare soil. The AI then writes a full '
        'environmental report explaining what those numbers mean for the environment.'
        '</div></div>',
        unsafe_allow_html=True
    )
    st.markdown("""
<div class="guide-grid">
  <div class="guide-chip">
    <strong>🌿 NDVI — Vegetation Health</strong>
    How green and dense the plants are. High = lush forest or healthy crops.
    Low = bare soil, desert, or urban surfaces.
    <div class="guide-scale">
      <span style="flex:0 0 auto;color:#6b7280;margin-right:4px;">-1</span>
      <div class="guide-scale-block" style="background:#dc2626;"></div>
      <div class="guide-scale-block" style="background:#f59e0b;"></div>
      <div class="guide-scale-block" style="background:#84cc16;"></div>
      <div class="guide-scale-block" style="background:#16a34a;"></div>
      <span style="flex:0 0 auto;color:#6b7280;margin-left:4px;">+1</span>
    </div>
    <div style="font-size:0.68rem;color:#6b7280;margin-top:3px;">
      &lt;0.1 = no plants &middot; 0.1&ndash;0.3 = sparse &middot; &gt;0.5 = dense forest
    </div>
  </div>
  <div class="guide-chip">
    <strong>💧 NDWI — Water Detection</strong>
    Detects open water: lakes, rivers, flooded fields.
    Positive values = water present. Negative = dry land.
    <div class="guide-scale">
      <span style="flex:0 0 auto;color:#6b7280;margin-right:4px;">-1</span>
      <div class="guide-scale-block" style="background:#d97706;"></div>
      <div class="guide-scale-block" style="background:#e5e7eb;"></div>
      <div class="guide-scale-block" style="background:#93c5fd;"></div>
      <div class="guide-scale-block" style="background:#1d4ed8;"></div>
      <span style="flex:0 0 auto;color:#6b7280;margin-left:4px;">+1</span>
    </div>
    <div style="font-size:0.68rem;color:#6b7280;margin-top:3px;">
      &gt;0.3 = open water &middot; near 0 = moist soil &middot; &lt;-0.2 = very dry
    </div>
  </div>
  <div class="guide-chip">
    <strong>🏙️ NDBI — Built-up Surfaces</strong>
    Highlights roads, rooftops, and concrete.
    High values = cities or industrial zones.
    <div class="guide-scale">
      <span style="flex:0 0 auto;color:#6b7280;margin-right:4px;">-1</span>
      <div class="guide-scale-block" style="background:#22c55e;"></div>
      <div class="guide-scale-block" style="background:#e5e7eb;"></div>
      <div class="guide-scale-block" style="background:#fcd34d;"></div>
      <div class="guide-scale-block" style="background:#f59e0b;"></div>
      <span style="flex:0 0 auto;color:#6b7280;margin-left:4px;">+1</span>
    </div>
    <div style="font-size:0.68rem;color:#6b7280;margin-top:3px;">
      negative = vegetation &middot; &gt;0.1 = urban/industrial
    </div>
  </div>
  <div class="guide-chip">
    <strong>📈 ΔNDVI — Vegetation Change</strong>
    Compares vegetation between two dates. Positive = more green over time.
    Negative = forest loss or land degradation.
    <div class="guide-scale">
      <span style="flex:0 0 auto;color:#6b7280;margin-right:4px;">-</span>
      <div class="guide-scale-block" style="background:#dc2626;"></div>
      <div class="guide-scale-block" style="background:#fca5a5;"></div>
      <div class="guide-scale-block" style="background:#bbf7d0;"></div>
      <div class="guide-scale-block" style="background:#16a34a;"></div>
      <span style="flex:0 0 auto;color:#6b7280;margin-left:4px;">+</span>
    </div>
    <div style="font-size:0.68rem;color:#6b7280;margin-top:3px;">
      red = vegetation loss &middot; green = vegetation gain
    </div>
  </div>
  <div class="guide-chip">
    <strong>🏜️ Aridity Index</strong>
    How dry the climate is over many years — rainfall vs evaporation.
    Low = desert. High = humid forest.
    <div style="font-size:0.68rem;color:#6b7280;margin-top:6px;">
      &lt;0.05 hyper-arid &middot; 0.05-0.2 arid &middot; 0.2-0.5 semi-arid &middot; &gt;0.65 humid
    </div>
  </div>
  <div class="guide-chip">
    <strong>🗺️ Land Cover Classes</strong>
    Every pixel is sorted into <em>Water</em>, <em>Vegetation</em>,
    <em>Urban</em>, or <em>Barren</em> land, shown as a percentage of the scene.
  </div>
  <div class="guide-chip">
    <strong>⚖️ Confidence Score</strong>
    How much data was available. Above 75% = findings well-supported.
    Below 55% = some data missing, interpret with caution.
  </div>
  <div class="guide-chip">
    <strong>🛰️ Data sources</strong>
    Images: Google Earth Engine (Sentinel-2 / Landsat).
    Climate: NASA POWER. AI report: Groq LLM.
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# Replace the generate_pdf_report function with this enhanced version:

def generate_pdf_report(ic, report_text, ndvi_map, ndwi_map, ndbi_map,
                        land_cover, delta, ndvi_t1_val, ndvi_t2_val,
                        t1_label, t2_label, delta_map, ts_df, ts_trend):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Image as RLImage, KeepTogether,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
        title="Geospatial Intelligence Report",
    )

    base   = getSampleStyleSheet()
    normal = base["Normal"]

    BLUE  = colors.HexColor("#2563eb")
    GRAY  = colors.HexColor("#dde3ed")
    GREEN = colors.HexColor("#16a34a")
    RED   = colors.HexColor("#dc2626")
    DARK  = colors.HexColor("#1a1f2e")

    def S(name, **kw):
        return ParagraphStyle(name, parent=normal, **kw)

    sTitle   = S("T",  fontSize=20, fontName="Helvetica-Bold",
                 textColor=DARK, spaceAfter=6, alignment=TA_CENTER)
    sSub     = S("Su", fontSize=10, textColor=colors.HexColor("#6b7280"),
                 spaceAfter=16, alignment=TA_CENTER)
    sH1      = S("H1", fontSize=14, fontName="Helvetica-Bold",
                 textColor=BLUE, spaceBefore=14, spaceAfter=6)
    sH2      = S("H2", fontSize=11, fontName="Helvetica-Bold",
                 textColor=DARK, spaceBefore=10, spaceAfter=4)
    sH3      = S("H3", fontSize=10, fontName="Helvetica-Bold",
                 textColor=colors.HexColor("#374151"), spaceBefore=8, spaceAfter=3)
    sBody    = S("Bo", fontSize=9, leading=14, spaceAfter=6, alignment=TA_JUSTIFY)
    sCaption = S("Ca", fontSize=7.5, textColor=colors.HexColor("#6b7280"),
                 spaceAfter=4, alignment=TA_CENTER)
    sMono    = S("Mo", fontSize=8, fontName="Courier",
                 textColor=DARK, spaceAfter=4)
    sLabel   = S("La", fontSize=7, fontName="Helvetica-Bold",
                 textColor=colors.HexColor("#6b7280"), spaceAfter=2)

    story = []

    # ==================== COVER PAGE ====================
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("Geospatial Intelligence Report", sTitle))
    region = getattr(ic, "region", None) or "Unknown region"
    ecosys = getattr(ic, "ecosystem", None) or "Unknown"
    t_cov  = (str(t1_label) + " to " + str(t2_label)) if t1_label and t2_label else "Single image"
    story.append(Paragraph(region + "  |  " + ecosys + "  |  " + t_cov, sSub))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=16))
    
    # Generation date
    from datetime import datetime
    story.append(Paragraph("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"), sCaption))
    
    # ==================== SECTION 1: OVERVIEW ====================
    story.append(PageBreak())
    story.append(Paragraph("1. Executive Overview", sH1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY, spaceAfter=8))
    
    # Region info
    story.append(Paragraph("📍 Location & Context", sH2))
    story.append(Paragraph(
        "<b>Region:</b> " + region + "<br/>"
        "<b>Ecosystem:</b> " + ecosys + "<br/>"
        "<b>Temporal Coverage:</b> " + t_cov,
        sBody
    ))
    story.append(Spacer(1, 0.3*cm))
    
    # Spectral Indices Table
    story.append(Paragraph("📊 Spectral Indices", sH2))
    
    conf   = getattr(ic, "confidence_score", 0) or 0
    ndvi_v = getattr(ic, "ndvi_mean", None)
    ndwi_v = getattr(ic, "ndwi_mean", None)
    ndbi_v = getattr(ic, "ndbi_mean", None)
    ai_v   = getattr(ic, "aridity_index", None)
    
    def fv(v): return "{:.3f}".format(v) if v is not None else "N/A"
    
    def interp_ndvi(v):
        if v is None: return "N/A"
        if v < 0.05:  return "Bare soil / water"
        if v < 0.12:  return "Very sparse vegetation"
        if v < 0.20:  return "Sparse vegetation"
        if v < 0.30:  return "Low-moderate vegetation"
        if v < 0.45:  return "Moderate vegetation"
        if v < 0.60:  return "Moderately high vegetation"
        if v < 0.75:  return "High vegetation"
        return "Dense forest"

    def interp_ndwi(v):
        if v is None:   return "N/A"
        if v > 0.30:    return "Permanent open water"
        if v > 0.10:    return "Shallow water / wetland"
        if v > 0.00:    return "Moist soil"
        if v > -0.15:   return "Dry surface"
        return "Very dry / salt crust"

    def interp_ndbi(v):
        if v is None:  return "N/A"
        if v > 0.20:   return "Dense urban"
        if v > 0.05:   return "Peri-urban / construction"
        if v > 0.00:   return "Sparse built-up"
        return "Vegetated / water dominated"

    def interp_ai(v):
        if v is None:  return "N/A"
        if v < 0.05:   return "Hyper-arid"
        if v < 0.20:   return "Arid"
        if v < 0.50:   return "Semi-arid"
        if v < 0.65:   return "Dry sub-humid"
        return "Humid"

    idx_data = [["Metric", "Value", "Interpretation"]]
    idx_data.append(["NDVI",          fv(ndvi_v),  interp_ndvi(ndvi_v)])
    idx_data.append(["NDWI",          fv(ndwi_v),  interp_ndwi(ndwi_v)])
    idx_data.append(["NDBI",          fv(ndbi_v),  interp_ndbi(ndbi_v)])
    idx_data.append(["Aridity Index",  fv(ai_v),   interp_ai(ai_v)])
    conf_label = "High" if conf >= 75 else "Moderate" if conf >= 55 else "Low"
    idx_data.append(["Confidence",    "{:.0f}%".format(conf), conf_label])
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        trend_lbl = "Vegetation gain" if delta > 0 else "Vegetation loss"
        idx_data.append(["Delta NDVI", sign + "{:.3f}".format(delta), trend_lbl])

    idx_table = Table(idx_data, colWidths=[3.5*cm, 2.5*cm, 9.5*cm])
    idx_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  BLUE),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID",          (0,0),(-1,-1), 0.4, GRAY),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(idx_table)
    story.append(Spacer(1, 0.4*cm))

    # Land Cover
    if land_cover:
        story.append(Paragraph("🌍 Land Cover Breakdown", sH2))
        lc_data = [["Class", "Coverage", "Visual"]]
        lc_colors_map = {
            "water": colors.HexColor("#3b82f6"),
            "vegetation": colors.HexColor("#22c55e"),
            "urban": colors.HexColor("#f59e0b"),
            "barren": colors.HexColor("#94a3b8"),
        }
        for cls, pct in land_cover.items():
            lc_data.append([cls.capitalize(), "{:.1f}%".format(pct), ""])
        lc_table = Table(lc_data, colWidths=[4*cm, 3*cm, 2*cm])
        lc_table.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,0), BLUE),
            ("TEXTCOLOR",    (0,0),(-1,0), colors.white),
            ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0),(-1,-1), 8),
            ("GRID",         (0,0),(-1,-1), 0.4, GRAY),
            ("LEFTPADDING",  (0,0),(-1,-1), 6),
            ("TOPPADDING",   (0,0),(-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        story.append(lc_table)
        story.append(Spacer(1, 0.3*cm))
    
    # Confidence Bar (text representation)
    story.append(Paragraph("📈 Confidence Assessment", sH2))
    conf_bar_len = int(conf / 5)
    conf_bar = "█" * conf_bar_len + "░" * (20 - conf_bar_len)
    story.append(Paragraph(
        f"<b>{conf:.0f}%</b> - {conf_label} confidence<br/>"
        f"<font face='Courier' size='8'>{conf_bar}</font>",
        sBody
    ))
    story.append(Spacer(1, 0.3*cm))
    
    # Image Metadata
    story.append(Paragraph("📷 Image Metadata", sH2))
    meta = ic.image_meta
    meta_parts = [
        "Format: " + str(getattr(ic, "image_format", "N/A")),
        "Bands: " + str(getattr(ic, "n_bands", "N/A")),
        "Dimensions: " + str(meta.get("width","?")) + " x " + str(meta.get("height","?")) + " px",
        "CRS: " + str(meta.get("crs","N/A")),
        "Sensor: " + str(meta.get("sensor","N/A")),
    ]
    story.append(Paragraph("  |  ".join(meta_parts), sMono))
    
    # Climate Summary
    cs = getattr(ic, "climate_summary", None)
    if cs:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("🌡️ Climate Snapshot", sH2))
        clim_parts = [
            "Rainfall (latest): " + "{:.1f}".format(cs.get("rainfall_mm_latest",0)) + " mm",
            "Trend: " + str(cs.get("rainfall_mm_trend","N/A")),
            "Temperature: " + "{:.1f}".format(cs.get("temperature_c_latest",0)) + " °C",
            "Humidity: " + "{:.1f}".format(cs.get("humidity_pct_latest",0)) + "%",
        ]
        story.append(Paragraph("  |  ".join(clim_parts), sMono))
    
    # Anomalies
    anomalies = getattr(ic, "anomalies", None)
    if anomalies:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("⚠️ Detected Anomalies", sH2))
        for a in anomalies:
            story.append(Paragraph("• " + a, sBody))
    
    # ==================== SECTION 2: INDEX MAPS ====================
    story.append(PageBreak())
    story.append(Paragraph("2. Spectral Index Maps", sH1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY, spaceAfter=8))
    
    maps_to_render = []
    if ndvi_map is not None: maps_to_render.append((ndvi_map, "NDVI - Vegetation Density", "RdYlGn"))
    if ndwi_map is not None: maps_to_render.append((ndwi_map, "NDWI - Water Content",      "Blues_r"))
    if ndbi_map is not None: maps_to_render.append((ndbi_map, "NDBI - Built-up Index",     "YlOrRd"))
    
    if maps_to_render:
        for idx, (arr, title, cmap) in enumerate(maps_to_render, 1):
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#ffffff")
            ax.set_facecolor("#f8fafc")
            im2 = ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            cbar = fig.colorbar(im2, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
            cbar.outline.set_edgecolor(GRAY)
            ax.set_title(title, fontsize=10, fontfamily="monospace", pad=6)
            ax.axis("off")
            fig.tight_layout(pad=0.4)
            
            ibuf = io.BytesIO()
            fig.savefig(ibuf, format="PNG", dpi=120, bbox_inches="tight")
            plt.close(fig)
            ibuf.seek(0)
            
            img_width = A4[0] - 4*cm
            img_height = img_width * 0.65
            story.append(RLImage(ibuf, width=img_width, height=img_height))
            story.append(Spacer(1, 0.2*cm))
            
            # Add interpretation caption
            if "NDVI" in title:
                story.append(Paragraph(
                    "<i>NDVI values range from -1 to +1. Higher values indicate healthier, denser vegetation.</i>",
                    sCaption
                ))
            elif "NDWI" in title:
                story.append(Paragraph(
                    "<i>NDWI highlights water bodies. Positive values (blue) indicate open water or high moisture.</i>",
                    sCaption
                ))
            elif "NDBI" in title:
                story.append(Paragraph(
                    "<i>NDBI identifies built-up areas. Higher values (yellow-red) indicate urban surfaces.</i>",
                    sCaption
                ))
            story.append(Spacer(1, 0.3*cm))
    
    # Temporal NDVI Comparison
    if delta is not None and ndvi_t1_val is not None and ndvi_t2_val is not None:
        story.append(Paragraph("📈 Temporal NDVI Comparison", sH2))
        sign2  = "+" if delta >= 0 else ""
        tlabel = "Vegetation gain" if delta > 0 else "Vegetation loss"
        tcol   = GREEN if delta >= 0 else RED
        
        t_data = [
            ["Date/Image", "NDVI Value", ""],
            [str(t1_label or "Image 1"), "{:.3f}".format(ndvi_t1_val), ""],
            [str(t2_label or "Image 2"), "{:.3f}".format(ndvi_t2_val), ""],
            ["<b>Change</b>", "<b>" + sign2 + "{:.3f}</b>".format(delta), "<b>" + tlabel + "</b>"],
        ]
        t_table = Table(t_data, colWidths=[5*cm, 4*cm, 6*cm])
        t_table.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,0), BLUE),
            ("TEXTCOLOR",    (0,0),(-1,0), colors.white),
            ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0),(-1,-1), 9),
            ("ALIGN",        (0,0),(-1,-1), "LEFT"),
            ("TEXTCOLOR",    (2,3),(2,3), tcol),
            ("FONTNAME",     (0,3),(-1,3), "Helvetica-Bold"),
            ("GRID",         (0,0),(-1,-1), 0.4, GRAY),
            ("TOPPADDING",   (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ]))
        story.append(t_table)
        story.append(Spacer(1, 0.3*cm))
    
    # Delta NDVI Map
    if delta_map is not None:
        story.append(Paragraph("🗺️ Delta NDVI Spatial Change Map", sH3))
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#ffffff")
        ax2.set_facecolor("#f8fafc")
        im3 = ax2.imshow(delta_map, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        cbar = fig2.colorbar(im3, ax=ax2, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_edgecolor(GRAY)
        ax2.set_title("ΔNDVI Change Map — Green: gain, Red: loss",
                      fontsize=9, fontfamily="monospace", pad=6)
        ax2.axis("off")
        fig2.tight_layout(pad=0.4)
        
        dm_buf = io.BytesIO()
        fig2.savefig(dm_buf, format="PNG", dpi=120, bbox_inches="tight")
        plt.close(fig2)
        dm_buf.seek(0)
        
        dw = A4[0] - 4*cm
        story.append(RLImage(dm_buf, width=dw, height=dw*0.55))
        story.append(Paragraph(
            "<i>Red areas indicate vegetation loss; green areas indicate vegetation gain over time.</i>",
            sCaption
        ))
        story.append(Spacer(1, 0.3*cm))
    
    # NDVI Time Series
    if ts_df is not None and len(ts_df) > 1:
        story.append(Paragraph("📊 NDVI Time Series (Annual)", sH2))
        try:
            from time_series import render_time_series_chart
            fig_ts = render_time_series_chart(
                ts_df, ts_trend or {},
                ecosystem=getattr(ic,"ecosystem","") or ""
            )
            ts_buf = io.BytesIO()
            fig_ts.savefig(ts_buf, format="PNG", dpi=120, bbox_inches="tight")
            plt.close(fig_ts)
            ts_buf.seek(0)
            tw = A4[0] - 4*cm
            story.append(RLImage(ts_buf, width=tw, height=tw*0.4))
            if ts_trend:
                tc = ts_trend
                story.append(Paragraph(
                    "Trend: " + str(tc.get("trend","N/A")) + "  |  " +
                    "Rate: " + "{:+.4f}".format(tc.get("annual_rate",0)) + "/yr  |  " +
                    "Total change: " + "{:+.3f}".format(tc.get("total_change",0)) + "  |  " +
                    "R²: " + "{:.2f}".format(tc.get("r2",0)),
                    sMono
                ))
        except Exception as e:
            story.append(Paragraph("Time series chart error: " + str(e), sBody))
    
    # ==================== SECTION 3: AI REPORT ====================
    story.append(PageBreak())
    story.append(Paragraph("3. AI-Generated Scientific Report", sH1))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))
    
    if report_text:
        # Clean up report text
        first = report_text.find("## 1.")
        if first == -1: first = report_text.find("## ")
        body = report_text[first:] if first != -1 else report_text
        fm = re.search(r'\n={10,}\s*\nConfidence:', body)
        if fm: body = body[:fm.start()].strip()
        else:  body = re.sub(r'\n={10,}.*$', '', body, flags=re.DOTALL).strip()
        
        sections = parse_report_sections(body)
        if sections:
            for title, content in sections.items():
                story.append(KeepTogether([
                    Paragraph(title, sH2),
                    HRFlowable(width="100%", thickness=0.4, color=GRAY, spaceAfter=4),
                ]))
                clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
                for line in clean.split("\n"):
                    line = line.strip()
                    if not line: continue
                    if line.startswith("* ") or line.startswith("- "):
                        story.append(Paragraph("• " + line[2:], sBody))
                    else:
                        story.append(Paragraph(line, sBody))
                story.append(Spacer(1, 0.2*cm))
        else:
            # Fallback if parsing fails
            clean_report = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', report_text)
            for para in clean_report.split("\n\n"):
                if para.strip():
                    story.append(Paragraph(para.strip(), sBody))
                    story.append(Spacer(1, 0.2*cm))
    else:
        story.append(Paragraph("Report not generated.", sBody))
    
    # ==================== FOOTER ====================
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY, spaceAfter=4))
    story.append(Paragraph(
        "Generated by Geospatial Intelligence Platform  |  " +
        "Data sources: Sentinel-2 / Landsat, NASA POWER  |  " +
        "Confidence: " + "{:.0f}".format(conf) + "%",
        sCaption
    ))
    
    # Build PDF
    doc.build(story)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="padding:0.5rem 0 1.2rem 0;">'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.05rem;color:#2563eb;font-weight:600;">🛰️ GeoIntel</div>'
        '<div style="font-size:0.70rem;color:#6b7280;margin-top:0.2rem;letter-spacing:0.05em;">MULTIMODAL GEOSPATIAL PLATFORM</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 📡 Image Input")
    input_mode = st.radio("Source", ["Upload image(s)", "Fetch from GEE"], horizontal=True)

    uploaded_t1 = uploaded_t2 = gee_params = None

    if input_mode == "Upload image(s)":
        uploaded_t1 = st.file_uploader("Primary image (or earlier date)",
                                        type=["tif","tiff","png","jpg"], key="img_t1")
        uploaded_t2 = st.file_uploader("Second image — optional (later date)",
                                        type=["tif","tiff","png","jpg"], key="img_t2")
        if uploaded_t1 is not None or uploaded_t2 is not None:
            st.markdown("#### 📅 Acquisition Dates")
            c1, c2 = st.columns(2)
            with c1: date_t1 = st.date_input("Image 1", value=None, key="date_t1")
            with c2: date_t2 = st.date_input("Image 2", value=None, key="date_t2")
        else:
            date_t1 = date_t2 = None
    else:
        st.markdown("**Location**")
        cl, co = st.columns(2)
        with cl: gee_lat = st.number_input("Latitude",  value=36.45, format="%.4f", key="gee_lat")
        with co: gee_lon = st.number_input("Longitude", value=10.73, format="%.4f", key="gee_lon")
        st.markdown("**Temporal range**")
        cy1, cy2 = st.columns(2)
        with cy1: gee_year1 = st.number_input("Year 1", value=2015, min_value=1984, max_value=2024, key="gee_y1")
        with cy2: gee_year2 = st.number_input("Year 2", value=2023, min_value=1984, max_value=2024, key="gee_y2")
        gee_sensor = st.selectbox("Sensor (auto if blank)",
                                   ["Auto","Sentinel-2 L2A","Landsat 8/9","Landsat 5 TM"],
                                   key="gee_sensor")
        gee_buffer = st.slider("Area radius (km)", 2.0, 20.0, 5.0, 0.5, key="gee_buf")
        gee_params = {
            "lat": gee_lat, "lon": gee_lon,
            "year1": int(gee_year1), "year2": int(gee_year2),
            "sensor": None if gee_sensor == "Auto" else gee_sensor,
            "buffer_km": gee_buffer,
        }
        date_t1 = _date(int(gee_year1), 4, 1)
        date_t2 = _date(int(gee_year2), 4, 1)

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Climate Data")
    climate_mode = st.radio("Source", ["Auto-fetch (NASA POWER)", "Upload CSV"], horizontal=True)
    uploaded_csv = None
    if climate_mode == "Upload CSV":
        uploaded_csv = st.file_uploader("NASA POWER CSV", type=["csv"], key="csv")

    fetch_timeseries = st.checkbox("📈 Fetch NDVI time series (GEE required)", value=False)

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Options")
    user_question = st.text_area(
        "Custom question (optional)",
        value="Provide a full scientific interpretation of this image and data.",
        height=80,
    )
    run_btn = st.button("▶ Run Analysis", use_container_width=True)

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.68rem;color:#9ca3af;line-height:1.7;">'
        'Pipeline: Image → ViT → RAG → LLM<br>'
        'LLM: Groq / llama-3.3-70b-versatile<br>'
        'Indices: NDVI · NDWI · NDBI · BSI · UI · MNDWI'
        '</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div style="margin-bottom:1.2rem;">'
    '<h1 style="margin:0;">Geospatial Intelligence Platform</h1>'
    '<div class="label-mono" style="margin-top:0.3rem;">'
    'Satellite imagery · Spectral analysis · AI-generated environmental reports'
    '</div></div>',
    unsafe_allow_html=True
)

if not run_btn:
    render_user_guide()
    st.markdown(
        '<div class="geo-card" style="text-align:center;padding:2.5rem 2rem;border-style:dashed;margin-top:0.5rem;">'
        '<div style="font-size:2.5rem;margin-bottom:1rem;">🛰️</div>'
        '<div style="font-family:\'IBM Plex Mono\',monospace;color:#2563eb;font-size:1rem;margin-bottom:0.5rem;">Ready for Analysis</div>'
        '<div style="color:#6b7280;font-size:0.85rem;max-width:420px;margin:0 auto;">'
        'Configure your image source in the sidebar, then click <strong>Run Analysis</strong>.'
        '</div></div>',
        unsafe_allow_html=True
    )
    st.stop()

if not uploaded_t1 and not gee_params:
    st.markdown(
        '<div class="geo-card geo-card-error">⚠️ <strong>No image source.</strong> '
        'Upload an image or configure GEE fetch in the sidebar.</div>',
        unsafe_allow_html=True
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

pipeline_warnings = []
status     = st.status("Running geospatial analysis pipeline…", expanded=True)
temp_files = []
meta_t1    = {}

try:
    with status:
        st.write("📦 Loading pipeline modules…")

        try:    from geospatial_platform.context       import InputContext
        except Exception as e: import_error_card("context->InputContext", e); st.stop()
        try:    from geospatial_platform.input_handler import build_input_context
        except Exception as e: import_error_card("input_handler->build_input_context", e); st.stop()
        try:    from geospatial_platform.input_handler import load_image
        except Exception as e: import_error_card("input_handler->load_image", e); st.stop()
        try:    from geospatial_platform.image_processor import process_image
        except Exception as e: import_error_card("image_processor->process_image", e); st.stop()
        try:    from geospatial_platform.vision_model  import extract_vit_features
        except Exception as e: import_error_card("vision_model->extract_vit_features", e); st.stop()

        build_climate_summary = populate_convenience_fields = None
        try:
            from geospatial_platform.data_integrator import (
                integrate_data, build_climate_summary, populate_convenience_fields,
            )
        except ImportError:
            try:    from geospatial_platform.data_integrator import integrate_data
            except Exception as e: import_error_card("data_integrator->integrate_data", e); st.stop()

        try:    from geospatial_platform.rag        import retrieve_context
        except Exception as e: import_error_card("rag->retrieve_context", e); st.stop()
        try:    from geospatial_platform.llm_engine import generate_report
        except Exception as e: import_error_card("llm_engine->generate_report", e); st.stop()

        # ── GEE fetch ─────────────────────────────────────────────────────────
        path_t1 = path_t2 = path_csv = None

        if gee_params:
            st.write("🛰️ Fetching imagery from Google Earth Engine…")
            try:
                from gee_connector import (
                    init_gee, fetch_image_as_array, save_array_as_geotiff,
                    auto_select_sensor, SENSOR_CONFIGS,
                )

                # Safe credential check
                st.write("🔍 Checking GEE credentials…")
                sa_info = st.secrets.get("GEE_SERVICE_ACCOUNT", None)
                if sa_info is None:
                    st.error("❌ GEE_SERVICE_ACCOUNT missing from Streamlit secrets."); st.stop()
                missing_keys = [k for k in ["project_id","private_key_id","private_key",
                                             "client_email","client_id"] if not sa_info.get(k)]
                if missing_keys:
                    st.error("❌ Missing secret fields: " + str(missing_keys)); st.stop()
                raw_key = sa_info.get("private_key", "")
                if "BEGIN PRIVATE KEY" not in raw_key:
                    st.error("❌ private_key is malformed — missing BEGIN PRIVATE KEY header.")
                    st.info("Tip: In Streamlit secrets, the private_key value must be a single "
                            "quoted string with \\n characters, not real line breaks.")
                    st.stop()
                st.write("  ✅ Service account: `" + str(sa_info.get("client_email")) + "`")
                st.write("  ✅ Project: `" + str(sa_info.get("project_id")) + "`")
                st.write("  ✅ Private key: present (" + str(len(raw_key)) + " chars)")

                st.write("🔐 Initialising GEE…")
                try:    gee_ok = init_gee()
                except Exception as e:
                    import traceback
                    st.error("❌ init_gee() raised: " + str(e))
                    st.code(traceback.format_exc()); st.stop()
                if not gee_ok:
                    st.error("❌ GEE initialisation failed. Check Streamlit Cloud logs for [GEE] print statements.")
                    st.info(
                        "Common causes:\n"
                        "1. private_key has wrong newline format\n"
                        "2. Earth Engine API not enabled in Google Cloud project\n"
                        "3. Service account lacks Earth Engine access\n"
                        "4. New key not yet propagated — wait 1-2 min and retry"
                    )
                    st.stop()
                st.write("✅ GEE initialized")

                sensor1 = gee_params.get("sensor") or auto_select_sensor(gee_params["year1"])
                st.write("Sensor: `" + sensor1 + "` | Year: `" + str(gee_params["year1"]) +
                         "` | Lat: `" + str(round(gee_params["lat"],4)) +
                         "` | Lon: `" + str(round(gee_params["lon"],4)) + "`")

                try:
                    import ee
                    point = ee.Geometry.Point([gee_params["lon"], gee_params["lat"]])
                    aoi   = point.buffer(gee_params["buffer_km"]*1000).bounds()
                    cfg   = SENSOR_CONFIGS[sensor1]
                    col   = (ee.ImageCollection(cfg["collection"])
                              .filterBounds(aoi)
                              .filterDate(str(gee_params["year1"])+"-01-01",
                                          str(gee_params["year1"])+"-12-31")
                              .filter(ee.Filter.lt(cfg["cloud_prop"], 80)))
                    count = col.size().getInfo()
                    st.write("📡 Scenes Year 1 (cloud<80%): `" + str(count) + "`")
                    if count == 0:
                        st.error("❌ No scenes found. Try a different year or larger buffer."); st.stop()
                except Exception as se:
                    st.warning("Scene count check failed (non-fatal): " + str(se))

                st.write("  Downloading image for `" + str(gee_params["year1"]) + "`…")
                result1 = fetch_image_as_array(
                    lat=gee_params["lat"], lon=gee_params["lon"],
                    year=gee_params["year1"], sensor=sensor1,
                    buffer_km=gee_params["buffer_km"],
                )
                if result1 is None:
                    st.error("❌ GEE image download failed. Check Streamlit Cloud logs for [GEE] traceback.")
                    st.stop()
                array_t1, meta_t1 = result1
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                    path_t1 = tmp.name
                path_t1 = save_array_as_geotiff(array_t1, meta_t1, path_t1)
                temp_files.append(path_t1)
                st.write("  ✓ Image 1 (`" + str(gee_params["year1"]) + "`) shape: `" + str(array_t1.shape) + "`")

                if gee_params["year2"] != gee_params["year1"]:
                    sensor2 = gee_params.get("sensor") or auto_select_sensor(gee_params["year2"])
                    st.write("  Downloading image for `" + str(gee_params["year2"]) + "`…")
                    result2 = fetch_image_as_array(
                        lat=gee_params["lat"], lon=gee_params["lon"],
                        year=gee_params["year2"], sensor=sensor2,
                        buffer_km=gee_params["buffer_km"],
                    )
                    if result2 is not None:
                        array_t2, meta_t2 = result2
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                            path_t2 = tmp.name
                        path_t2 = save_array_as_geotiff(array_t2, meta_t2, path_t2)
                        temp_files.append(path_t2)
                        st.write("  ✓ Image 2 (`" + str(gee_params["year2"]) + "`) shape: `" + str(array_t2.shape) + "`")
                    else:
                        pipeline_warnings.append("Image 2 fetch failed — temporal analysis skipped.")

            except Exception as e:
                import traceback
                st.error("❌ GEE error: " + str(e)); st.code(traceback.format_exc()); st.stop()
        else:
            st.write("📥 Saving uploaded files…")
            if uploaded_t1:
                path_t1 = save_upload_to_temp(uploaded_t1); temp_files.append(path_t1)
            if uploaded_t2:
                path_t2 = save_upload_to_temp(uploaded_t2); temp_files.append(path_t2)

        if uploaded_csv:
            path_csv = save_upload_to_temp(uploaded_csv); temp_files.append(path_csv)

        st.write("📥 Validating inputs…")
        ic = build_input_context(image_path=path_t1, csv_path=path_csv, question=user_question)

        if gee_params:
            ic.lat = gee_params["lat"]; ic.lon = gee_params["lon"]
            if meta_t1.get("region_name"):
                ic.image_meta["region_name"] = meta_t1["region_name"]
                ic.region = meta_t1["region_name"]
            if meta_t1.get("crs"):
                ic.image_meta["crs"] = meta_t1["crs"]

        if date_t1 is not None: ic.acquisition_month = date_t1.month
        if date_t2 is not None: ic.acquisition_month_t2 = date_t2.month

        if path_t2:
            st.write("📅 Loading second image…")
            arr_t2_l, meta_t2_l, _, _ = load_image(path_t2)
            ic.image_array_t2 = arr_t2_l; ic.image_meta_t2 = meta_t2_l

        st.write("🔬 Computing spectral indices…")
        ic = process_image(ic)

        if path_t2 and getattr(ic, "image_array_t2", None) is not None:
            st.write("📅 Computing temporal NDVI delta…")
            from geospatial_platform.image_processor import (
                compute_ndvi, detect_sensor, BAND_CONFIG,
                is_prescaled, get_band, normalize_to_reflectance,
            )
            arr2 = ic.image_array_t2
            cfg2 = BAND_CONFIG[detect_sensor(arr2.shape[0])]
            pre2 = is_prescaled(arr2)
            r2   = get_band(arr2, cfg2, "red"); n2 = get_band(arr2, cfg2, "nir")
            if r2 is not None and n2 is not None:
                r2 = normalize_to_reflectance(r2, pre2)
                n2 = normalize_to_reflectance(n2, pre2)
            ndvi_t2 = compute_ndvi(r2, n2)
            if ndvi_t2 is not None and not np.all(np.isnan(ndvi_t2)):
                mt1 = float(np.nanmean(ic.ndvi)) if ic.ndvi is not None else None
                mt2 = float(np.nanmean(ndvi_t2))
                ic.ndvi_mean_t1 = mt1; ic.ndvi_mean_t2 = mt2
                ic.ndvi_delta   = round(mt2 - mt1, 4) if mt1 is not None else None
                def _yfn(fname):
                    m = re.search(r'(19|20)\d{2}', fname or ''); return m.group(0) if m else None
                ic.temporal_label_t1 = (_yfn(uploaded_t1.name) if uploaded_t1
                                         else str(gee_params["year1"]) if gee_params else "Image 1")
                ic.temporal_label_t2 = (_yfn(uploaded_t2.name) if uploaded_t2
                                         else str(gee_params["year2"]) if gee_params else "Image 2")
                if ic.ndvi_delta is not None and abs(ic.ndvi_delta) > 0.05:
                    direction = "improvement" if ic.ndvi_delta > 0 else "decline"
                    ic.anomalies = list(ic.anomalies or [])
                    ic.anomalies.append("vegetation " + direction + " detected (NDVI=" + "{:+.3f}".format(ic.ndvi_delta) + ")")

        st.write("🧠 Extracting Vision Transformer features…")
        ic = extract_vit_features(ic)

        if path_csv:
            st.write("📊 Integrating uploaded climate CSV…")
            try:
                ic = integrate_data(ic)
                df = getattr(ic, "csv_df", None)
                if df is not None and build_climate_summary:
                    ic.climate_summary = build_climate_summary(df)
                    if populate_convenience_fields: populate_convenience_fields(ic)
            except Exception as e: pipeline_warnings.append("Climate integration failed: " + str(e))
        elif climate_mode == "Auto-fetch (NASA POWER)":
            st.write("🌍 Auto-fetching climate data from NASA POWER…")
            try:
                from climate_fetcher import fetch_nasa_power, climate_data_quality_report
                flat = gee_params["lat"] if gee_params else _extract_lat_from_meta(ic.image_meta)
                flon = gee_params["lon"] if gee_params else _extract_lon_from_meta(ic.image_meta)
                if flat is not None and flon is not None:
                    cdf = fetch_nasa_power(flat, flon)
                    if cdf is not None:
                        qr = climate_data_quality_report(cdf)
                        if not qr["suitable"]:
                            pipeline_warnings.append("NASA POWER: " + str(qr["n_years"]) + " yrs, " + str(qr["missing_rain"]) + " missing.")
                        ic.csv_df = cdf; ic = integrate_data(ic)
                        if build_climate_summary: ic.climate_summary = build_climate_summary(ic.csv_df)
                        if populate_convenience_fields: populate_convenience_fields(ic)
                        st.write("  ✓ " + str(qr["n_years"]) + " years of climate data fetched")
                    else: pipeline_warnings.append("NASA POWER returned no data.")
                else: pipeline_warnings.append("Coordinates unavailable for NASA POWER.")
            except Exception as e: pipeline_warnings.append("NASA POWER failed: " + str(e))
        else:
            st.write("📊 No climate data — skipping.")

        st.write("📚 Retrieving environmental context (RAG)…")
        ic = retrieve_context(ic)
        if not getattr(ic, "rag_context", None):
            ic.rag_context = getattr(ic, "retrieved_context", "") or ""

        ic.ndvi_timeseries = ic.ndvi_trend_stats = None
        if fetch_timeseries:
            st.write("📈 Fetching NDVI time series from GEE…")
            try:
                from time_series import fetch_ndvi_time_series, compute_trend, estimate_growing_season_months
                from gee_connector import init_gee
                if init_gee():
                    tlat = gee_params["lat"] if gee_params else _extract_lat_from_meta(ic.image_meta)
                    tlon = gee_params["lon"] if gee_params else _extract_lon_from_meta(ic.image_meta)
                    if tlat and tlon:
                        ms, me = estimate_growing_season_months(tlat, ic.aridity_index)
                        tsdf = fetch_ndvi_time_series(tlat, tlon, start_year=2010, end_year=2024,
                                                       month_start=ms, month_end=me)
                        if tsdf is not None:
                            ic.ndvi_timeseries = tsdf; ic.ndvi_trend_stats = compute_trend(tsdf)
                            st.write("  ✓ " + str(len(tsdf)) + " years of NDVI data")
                        else: pipeline_warnings.append("NDVI time series returned no data.")
                    else: pipeline_warnings.append("Coordinates unavailable for time series.")
                else: pipeline_warnings.append("GEE not initialised — time series skipped.")
            except Exception as e: pipeline_warnings.append("Time series failed: " + str(e))

        for attr in ("ndvi", "ndwi", "ndbi"):
            arr = getattr(ic, attr, None)
            if arr is not None and getattr(ic, attr+"_mean", None) is None:
                setattr(ic, attr+"_mean", float(np.nanmean(arr)))
                setattr(ic, attr+"_map",  arr)

        if not getattr(ic,"region",None):    ic.region    = ic.image_meta.get("region_name","Unknown region")
        if not getattr(ic,"ecosystem",None): ic.ecosystem = ic.image_meta.get("ecosystem","Mixed landscape")

        if ic.confidence_score is None:
            s = 0.0
            if ic.ndvi is not None:                      s += 20
            if ic.ndwi is not None:                      s += 15
            if ic.ndbi is not None:                      s += 10
            if ic.aridity_index is not None:             s += 10
            if getattr(ic,"csv_df",None) is not None:    s += 20
            if ic.ndvi_delta is not None:                s += 15
            ic.confidence_score = min(85.0, s)

        st.write("✍️ Generating scientific report…")
        ic.report = generate_report(ic, ic.rag_context or "", ic.anomalies or [])

    status.update(label="✅ Analysis complete", state="complete", expanded=False)
    for w in pipeline_warnings: st.warning(w)

except Exception as e:
    status.update(label="❌ Pipeline error", state="error", expanded=True)
    st.markdown(
        '<div class="geo-card geo-card-error"><strong>Pipeline failed:</strong><br><br>'
        '<code style="font-size:0.78rem;">' + type(e).__name__ + ': ' + str(e) + '</code></div>',
        unsafe_allow_html=True
    )
    st.exception(e); st.stop()
finally:
    for p in temp_files:
        try: os.unlink(p)
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# SHARED RESULT VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

region_str  = getattr(ic,"region",None) or ic.image_meta.get("region_name","Unknown region")
eco_str     = getattr(ic,"ecosystem",None) or "—"
t1_label    = getattr(ic,"temporal_label_t1",None)
t2_label    = getattr(ic,"temporal_label_t2",None)
temporal_str= (str(t1_label) + " → " + str(t2_label)) if t1_label and t2_label else "Single image"
ndvi        = getattr(ic,"ndvi_mean",None)
ndwi        = getattr(ic,"ndwi_mean",None)
ndbi        = getattr(ic,"ndbi_mean",None)
ai_val      = getattr(ic,"aridity_index",None)
delta       = getattr(ic,"ndvi_delta",None)
ndvi_t1_val = getattr(ic,"ndvi_mean_t1",None)
ndvi_t2_val = getattr(ic,"ndvi_mean_t2",None)
score       = getattr(ic,"confidence_score",None) or 0
land_cover  = getattr(ic,"land_cover",None)
anomalies   = getattr(ic,"anomalies",None)
ndvi_map    = getattr(ic,"ndvi_map",None)
ndwi_map    = getattr(ic,"ndwi_map",None)
ndbi_map    = getattr(ic,"ndbi_map",None)
delta_map   = getattr(ic,"ndvi_trend_map",None)
ts_df       = getattr(ic,"ndvi_timeseries",None)
ts_trend    = getattr(ic,"ndvi_trend_stats",None)
report      = getattr(ic,"report",None)
climate_sum = getattr(ic,"climate_summary",None)
lc_colors   = {"water":"#3b82f6","vegetation":"#22c55e","urban":"#f59e0b","barren":"#94a3b8"}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED RENDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _render_overview_content():
    st.markdown(
        '<div class="geo-card geo-card-accent">'
        '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">'
        '<div><div class="label-mono">Region</div>'
        '<div style="font-size:1.05rem;font-weight:600;color:#1a1f2e;margin-top:0.2rem;">📍 ' + region_str + '</div>'
        '<div style="font-size:0.8rem;color:#6b7280;margin-top:0.2rem;">' + eco_str + '</div></div>'
        '<div style="text-align:right;"><div class="label-mono">Temporal Coverage</div>'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.9rem;color:#2563eb;margin-top:0.2rem;">📅 ' + temporal_str + '</div>'
        '</div></div></div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("## Spectral Indices")
        chips = ""
        if ndvi   is not None: chips += '<div class="metric-chip">NDVI<span>'    + "{:.3f}".format(ndvi)   + '</span></div>'
        if ndwi   is not None: chips += '<div class="metric-chip">NDWI<span>'    + "{:.3f}".format(ndwi)   + '</span></div>'
        if ndbi   is not None: chips += '<div class="metric-chip">NDBI<span>'    + "{:.3f}".format(ndbi)   + '</span></div>'
        if ai_val is not None: chips += '<div class="metric-chip">Aridity<span>' + "{:.3f}".format(ai_val) + '</span></div>'
        if delta  is not None:
            sign = "+" if delta >= 0 else ""
            dc   = "#16a34a" if delta >= 0 else "#dc2626"
            chips += '<div class="metric-chip">ΔNDVI<span style="color:' + dc + '">' + sign + "{:.3f}".format(delta) + '</span></div>'
        st.markdown('<div class="metric-row">' + chips + '</div>', unsafe_allow_html=True)

        if land_cover:
            st.markdown(
                '<div style="font-family:IBM Plex Mono,monospace;font-size:1.05rem;font-weight:600;'
                'color:#2563eb;border-bottom:1px solid #dde3ed;padding-bottom:0.4rem;margin:1rem 0 0.6rem 0;">'
                'Land Cover</div>',
                unsafe_allow_html=True
            )
            st.markdown(render_land_cover_bar(land_cover), unsafe_allow_html=True)
            lcc = st.columns(4)
            for i, (cls, pct) in enumerate(land_cover.items()):
                with lcc[i % 4]:
                    c = lc_colors.get(cls, "#94a3b8")
                    st.markdown(
                        '<div style="text-align:center;margin-top:0.5rem;">'
                        '<div style="width:12px;height:12px;background:' + c + ';border-radius:2px;margin:0 auto 3px;"></div>'
                        '<div class="label-mono">' + cls + '</div>'
                        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1rem;color:#1a1f2e;font-weight:600;">'
                        + "{:.1f}".format(pct) + '%</div></div>',
                        unsafe_allow_html=True
                    )

        if anomalies:
            st.markdown("## Detected Anomalies")
            tags = "".join('<span class="anomaly-tag">⚠ ' + a + '</span>' for a in anomalies)
            st.markdown('<div>' + tags + '</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("## Confidence Score")
        lc_c = "#16a34a" if score>=75 else "#d97706" if score>=55 else "#dc2626"
        lt   = "High"    if score>=75 else "Moderate" if score>=55 else "Low"
        st.markdown(
            '<div class="geo-card">'
            '<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:2rem;color:' + lc_c + ';font-weight:600;">'
            + "{:.0f}".format(score) + '%</div>'
            '<div class="label-mono">' + lt + ' confidence</div></div>'
            + render_confidence_bar(score) + '</div>',
            unsafe_allow_html=True
        )

        st.markdown("## Image Metadata")
        meta = ic.image_meta
        mitems = [
            ("Format",     getattr(ic,"image_format","—")),
            ("Bands",      str(getattr(ic,"n_bands","—"))),
            ("Dimensions", str(meta.get("width","?")) + " × " + str(meta.get("height","?")) + " px"),
            ("CRS",        meta.get("crs","—")),
            ("Context",    meta.get("region_context","—")),
        ]
        rows = "".join(
            '<div class="meta-row">'
            '<span class="label-mono">' + k + '</span>'
            '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;color:#374151;text-align:right;max-width:55%;">' + str(v) + '</span>'
            '</div>'
            for k, v in mitems
        )
        st.markdown('<div class="geo-card">' + rows + '</div>', unsafe_allow_html=True)

        if climate_sum:
            st.markdown("## Climate Snapshot")
            cpairs = [
                ("Rainfall (latest)", "{:.1f}".format(climate_sum.get("rainfall_mm_latest",0)) + " mm"),
                ("Rainfall trend",    str(climate_sum.get("rainfall_mm_trend","—")).capitalize()),
                ("Temperature",       "{:.1f}".format(climate_sum.get("temperature_c_latest",0)) + " °C"),
                ("Humidity",          "{:.1f}".format(climate_sum.get("humidity_pct_latest",0)) + " %"),
            ]
            rows = "".join(
                '<div class="meta-row">'
                '<span class="label-mono">' + k + '</span>'
                '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;color:#374151;">' + v + '</span>'
                '</div>'
                for k, v in cpairs
            )
            st.markdown('<div class="geo-card">' + rows + '</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="geo-card" style="text-align:center;color:#9ca3af;font-size:0.8rem;padding:1.5rem;">'
                'No climate data uploaded.</div>',
                unsafe_allow_html=True
            )


def _render_index_maps_content():
    st.markdown("## Spectral Index Maps")
    available = []
    if ndvi_map is not None: available.append((ndvi_map, "NDVI — Vegetation Density", "RdYlGn"))
    if ndwi_map is not None: available.append((ndwi_map, "NDWI — Water Content",      "Blues_r"))
    if ndbi_map is not None: available.append((ndbi_map, "NDBI — Built-up Index",     "YlOrRd"))

    if not available:
        st.markdown('<div class="geo-card" style="text-align:center;color:#9ca3af;padding:2rem;">No index maps available.</div>', unsafe_allow_html=True)
    else:
        cols = st.columns(len(available))
        for col, (arr, title, cmap) in zip(cols, available):
            with col:
                fig = render_index_map(arr, title, cmap)
                st.pyplot(fig, use_container_width=True); plt.close(fig)

    if delta is not None and ndvi_t1_val is not None and ndvi_t2_val is not None:
        st.markdown("## Temporal NDVI Comparison")
        card_cls    = "geo-card-good" if delta >= 0 else "geo-card-error"
        arrow       = "↑ +" + "{:.3f}".format(delta) if delta >= 0 else "↓ " + "{:.3f}".format(delta)
        arrow_color = "#16a34a" if delta >= 0 else "#dc2626"
        st.markdown(
            '<div class="geo-card ' + card_cls + '">'
            '<div style="display:flex;gap:2.5rem;align-items:center;flex-wrap:wrap;">'
            '<div><div class="label-mono">' + str(t1_label or "Image 1") + '</div>'
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.4rem;color:#1a1f2e;">' + "{:.3f}".format(ndvi_t1_val) + '</div></div>'
            '<div style="font-size:1.4rem;color:#9ca3af;">→</div>'
            '<div><div class="label-mono">' + str(t2_label or "Image 2") + '</div>'
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.4rem;color:#1a1f2e;">' + "{:.3f}".format(ndvi_t2_val) + '</div></div>'
            '<div style="margin-left:1rem;"><div class="label-mono">Change (ΔNDVI)</div>'
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.6rem;color:' + arrow_color + ';font-weight:700;">' + arrow + '</div>'
            '</div></div></div>',
            unsafe_allow_html=True
        )

    if delta_map is not None:
        st.markdown("## ΔNDVI Spatial Change Map")
        st.markdown('<div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.5rem;">Green = vegetation gain · Red = vegetation loss</div>', unsafe_allow_html=True)
        fig_d = render_index_map(delta_map, "ΔNDVI Change Map (pixel-wise)", "RdYlGn")
        st.pyplot(fig_d, use_container_width=True); plt.close(fig_d)

    if ts_df is not None and len(ts_df) > 1:
        st.markdown("## NDVI Time Series (Annual)")
        try:
            from time_series import render_time_series_chart
            fig_ts = render_time_series_chart(ts_df, ts_trend or {}, ecosystem=getattr(ic,"ecosystem","") or "")
            st.pyplot(fig_ts, use_container_width=True); plt.close(fig_ts)
            if ts_trend:
                tc = "#16a34a" if (ts_trend.get("slope") or 0) > 0 else "#dc2626"
                st.markdown(
                    '<div class="metric-row">'
                    '<div class="metric-chip">Trend<span>' + str(ts_trend.get("trend","—")) + '</span></div>'
                    '<div class="metric-chip">Rate<span style="color:' + tc + '">' + "{:+.4f}".format(ts_trend.get("annual_rate",0)) + '/yr</span></div>'
                    '<div class="metric-chip">Total Δ<span style="color:' + tc + '">' + "{:+.3f}".format(ts_trend.get("total_change",0)) + '</span></div>'
                    '<div class="metric-chip">R²<span>' + "{:.2f}".format(ts_trend.get("r2",0)) + '</span></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.caption("Time series chart error: " + str(e))


def _render_ai_report():
    if not report:
        st.markdown('<div class="geo-card" style="text-align:center;color:#9ca3af;padding:2rem;">Report not generated.</div>', unsafe_allow_html=True)
        return
    first_s = report.find("## 1.")
    if first_s == -1: first_s = report.find("## ")
    report_body = report[first_s:] if first_s != -1 else report
    fm = re.search(r'\n={10,}\s*\nConfidence:', report_body)
    if fm: report_body = report_body[:fm.start()].strip()
    else:  report_body = re.sub(r'\n={10,}.*$', '', report_body, flags=re.DOTALL).strip()
    sections = parse_report_sections(report_body)
    if sections:
        for title, content in sections.items():
            render_report_section(title, content, get_icon(title))
    else:
        st.markdown('<div class="report-section" style="white-space:pre-wrap;">' + report + '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD — THREE TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_maps, tab_report = st.tabs([
    "  📊 Overview  ", "  🗺️ Index Maps  ", "  📄 Full Report  ",
])

with tab_overview:
    _render_overview_content()

with tab_maps:
    _render_index_maps_content()

with tab_report:
    st.markdown("## Full Analysis Report")

    # Overview summary
    _render_overview_content()
    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)

    # Index maps
    _render_index_maps_content()
    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)

    # AI report
    st.markdown("## AI-Generated Scientific Report")
    _render_ai_report()

    st.markdown("<hr class='geo-divider'>", unsafe_allow_html=True)

    # Downloads
    st.markdown("## Download Report")
    dl_col1, dl_col2, _ = st.columns([1, 1, 2])

    with dl_col1:
        if report:
            st.download_button(
                label="⬇ Download TXT",
                data=report,
                file_name="geointel_" + region_str.replace(" ","_").replace(",","") + ".txt",
                mime="text/plain",
            )

    with dl_col2:
        try:
            pdf_bytes = generate_pdf_report(
                ic, report, ndvi_map, ndwi_map, ndbi_map,
                land_cover, delta, ndvi_t1_val, ndvi_t2_val,
                t1_label, t2_label, delta_map, ts_df, ts_trend,
            )
            st.download_button(
                label="⬇ Download PDF",
                data=pdf_bytes,
                file_name="geointel_" + region_str.replace(" ","_").replace(",","") + ".pdf",
                mime="application/pdf",
            )
        except Exception as pdf_err:
            st.caption("PDF generation failed: " + str(pdf_err) + " — install reportlab to enable PDF export.")
