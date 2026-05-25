import os
from groq import Groq
from dataclasses import dataclass
from typing import Optional
import numpy as np


# ── Groq client ──────────────────────────────────────────────────────────────

def get_groq_client():
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=api_key)


# ── Contextual interpretation helpers ────────────────────────────────────────

def interpret_ndvi(ndvi: float) -> str:
    if ndvi is None:
        return "not available"
    if ndvi < 0.1:
        return "near-zero (bare soil or rock, no meaningful vegetation)"
    elif ndvi < 0.2:
        return "very low (sparse or stressed vegetation, typical of arid zones)"
    elif ndvi < 0.35:
        return "low-moderate (open shrubland or degraded vegetation cover)"
    elif ndvi < 0.5:
        return "moderate (mixed shrub-grassland or seasonally active vegetation)"
    elif ndvi < 0.65:
        return "moderately high (dense shrubland or productive cropland)"
    else:
        return "high (dense, healthy canopy — forest or irrigated agriculture)"


def interpret_ndwi(ndwi: float) -> str:
    if ndwi is None:
        return "not available"
    if ndwi > 0.3:
        return "strongly positive — open water body clearly present"
    elif ndwi > 0.0:
        return "mildly positive — wet soils or shallow water influence"
    elif ndwi > -0.15:
        return "near-zero — dry surface or moist but non-flooded land"
    else:
        return "negative — dry soil or vegetation with no surface water signal"


def interpret_ndbi(ndbi: float) -> str:
    if ndbi is None:
        return "not available"
    if ndbi > 0.2:
        return "high — significant built-up or impervious surface"
    elif ndbi > 0.0:
        return "low-moderate — sparse urban features or bare rock"
    else:
        return "negative — vegetated or water-dominated surface, minimal built-up area"


def interpret_aridity(ai: float) -> str:
    """UNESCO aridity classes with ecological meaning."""
    if ai is None:
        return "not available"
    if ai < 0.05:
        return "Hyper-arid (desert core; essentially no rain-fed vegetation possible)"
    elif ai < 0.2:
        return "Arid (sparse xerophytic vegetation; agriculture requires irrigation)"
    elif ai < 0.5:
        return "Semi-arid (steppe ecosystems; rain-fed agriculture marginal and risky)"
    elif ai < 0.65:
        return "Dry sub-humid (Mediterranean-type; seasonal drought stress common)"
    else:
        return "Humid (sufficient rainfall for dense vegetation without irrigation)"


def interpret_ndvi_delta(delta: float) -> str:
    if delta is None:
        return "no temporal data"
    if delta > 0.15:
        return (
            f"large positive shift (+{delta:.3f}) — substantial vegetation increase "
            "consistent with land use change, reforestation, or a multi-year wet period"
        )
    elif delta > 0.05:
        return (
            f"moderate positive shift (+{delta:.3f}) — gradual greening, "
            "possibly linked to improved rainfall or reduced grazing pressure"
        )
    elif delta > -0.05:
        return f"stable (±{delta:.3f}) — no meaningful change in vegetation density"
    elif delta > -0.15:
        return (
            f"moderate negative shift ({delta:.3f}) — vegetation decline, "
            "possible drought stress, overgrazing, or land clearing"
        )
    else:
        return (
            f"large negative shift ({delta:.3f}) — severe vegetation loss, "
            "consistent with desertification, fire, or major land use change"
        )


def get_regional_context(ecosystem: str, region: str) -> str:
    """Inject region-specific baseline knowledge into the prompt."""
    base = ""
    eco_lower = ecosystem.lower() if ecosystem else ""

    if "mediterranean" in eco_lower:
        base = (
            "Mediterranean ecosystems are characterised by hot dry summers and mild wet winters "
            "(Köppen Csa/Csb). Vegetation follows a strong seasonal pulse: NDVI peaks in late winter "
            "to spring (Feb–May) and drops sharply in summer. Annual rainfall in coastal Tunisia "
            "typically ranges 400–600 mm concentrated in Oct–Mar. "
            "An aridity index below 0.5 in this biome indicates significant dry-season stress "
            "even when annual totals appear adequate. "
            "A 26%+ water fraction in a Mediterranean coastal scene is most consistent with "
            "a lagoon, sebkha (salt flat), or wetland — not active flooding, unless the image "
            "was acquired after an extreme rainfall event."
        )
    elif "arid" in eco_lower or "semi-arid" in eco_lower:
        base = (
            "Arid and semi-arid ecosystems have extremely variable interannual rainfall. "
            "Vegetation cover is sparse and highly responsive to rainfall pulses; "
            "NDVI can spike after a single wet season and recede quickly. "
            "Barren fractions above 60% are climatically normal in these zones."
        )
    else:
        base = (
            "No specific ecosystem baseline loaded. Interpret indices relative to global norms."
        )

    if region:
        base += f" Analysis region: {region}."

    return base


# ── Contradiction detector ────────────────────────────────────────────────────

def detect_contradictions(context: dict) -> list[str]:
    """
    Identify contradictions in the data before passing to LLM,
    so the prompt can explicitly ask the model to resolve them.
    """
    contradictions = []

    aridity = context.get("aridity_index")
    humidity = context.get("humidity_pct")
    rainfall_trend = context.get("rainfall_trend", "")
    water_pct = context.get("water_pct", 0)
    ndwi = context.get("ndwi_mean", 0)

    # Arid classification but high humidity
    if aridity is not None and aridity < 0.2 and humidity is not None and humidity > 65:
        contradictions.append(
            f"Aridity index ({aridity:.3f}) classifies the area as Arid, "
            f"yet current humidity is {humidity:.1f}% — above the regional mean. "
            "This tension likely reflects seasonal timing: the image or climate record "
            "may have been acquired during or shortly after the wet season."
        )

    # High water fraction but negative NDWI
    if water_pct > 20 and ndwi is not None and ndwi < 0:
        contradictions.append(
            f"Land cover shows {water_pct:.1f}% water fraction, "
            f"but NDWI mean is {ndwi:.3f} (negative). "
            "This is consistent with a spectrally mixed coastal feature such as a sebkha or salt flat, "
            "where reflectance differs from open freshwater. Flooding is unlikely."
        )

    # Increasing rainfall trend but arid classification
    if aridity is not None and aridity < 0.2 and "increasing" in rainfall_trend:
        contradictions.append(
            f"Rainfall trend is increasing, yet the long-term aridity index ({aridity:.3f}) "
            "remains in the Arid range. This may indicate the recent wet trend is short-term "
            "and insufficient to shift the multi-decadal water balance."
        )

    return contradictions


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(input_context, rag_context: str, anomalies: list[str]) -> str:
    """
    Build the LLM prompt. Data is pre-interpreted before injection
    so the LLM reasons about meaning, not raw numbers.
    """

    ic = input_context

    # --- Pre-interpret all indices ---
    ndvi_interp   = interpret_ndvi(ic.ndvi_mean) if ic.ndvi_mean is not None else "not available"
    ndwi_interp   = interpret_ndwi(ic.ndwi_mean) if ic.ndwi_mean is not None else "not available"
    ndbi_interp   = interpret_ndbi(ic.ndbi_mean) if ic.ndbi_mean is not None else "not available"
    aridity_interp = interpret_aridity(ic.aridity_index) if ic.aridity_index is not None else "not available"
    delta_interp  = interpret_ndvi_delta(ic.ndvi_delta) if ic.ndvi_delta is not None else "no temporal data"

    # --- Build contradiction list ---
    ctx_dict = {
        "aridity_index": ic.aridity_index,
        "humidity_pct":  ic.humidity_pct,
        "rainfall_trend": ic.rainfall_trend or "",
        "water_pct":     ic.water_pct,
        "ndwi_mean":     ic.ndwi_mean,
    }
    contradictions = detect_contradictions(ctx_dict)
    contradiction_block = ""
    if contradictions:
        contradiction_block = "\n\nIDENTIFIED DATA CONTRADICTIONS (you MUST resolve each one):\n"
        for i, c in enumerate(contradictions, 1):
            contradiction_block += f"  {i}. {c}\n"

    # --- Regional context ---
    regional_ctx = get_regional_context(ic.ecosystem, ic.region)

    # --- Land cover summary ---
    lc_lines = ""
    if ic.land_cover:
        for cls, pct in ic.land_cover.items():
            lc_lines += f"    {cls:<14}: {pct:.1f}%\n"
    else:
        lc_lines = "    No land cover data available\n"

    # --- Climate summary (interpreted, not raw) with safe None handling ---
    climate_block = ""
    if ic.climate_summary:
        cs = ic.climate_summary
        rf_latest = cs.get("rainfall_mm_latest")
        rf_mean   = cs.get("rainfall_mm_mean")
        rf_trend  = cs.get("rainfall_mm_trend", "unknown")
        t_latest  = cs.get("temperature_c_latest")
        t_mean    = cs.get("temperature_c_mean")
        hum       = cs.get("humidity_pct_latest")
        cv        = cs.get("rainfall_cv")
        
        # Safe formatting with None checks
        rf_latest_str = f"{rf_latest:.1f}" if rf_latest is not None else "N/A"
        rf_mean_str = f"{rf_mean:.1f}" if rf_mean is not None else "N/A"
        t_latest_str = f"{t_latest:.1f}" if t_latest is not None else "N/A"
        t_mean_str = f"{t_mean:.1f}" if t_mean is not None else "N/A"
        hum_str = f"{hum:.1f}" if hum is not None else "N/A"
        cv_str = f"{cv:.2f}" if cv is not None else "N/A"
        
        # Determine seasonality text
        if cv is not None:
            if cv > 0.7:
                seasonality_text = "very high seasonality; single-month rainfall values are poor indicators of annual conditions"
            else:
                seasonality_text = "moderate seasonality"
        else:
            seasonality_text = "insufficient data to determine seasonality"

        climate_block = f"""
CLIMATE DATA (15-year record):
  Rainfall  : latest {rf_latest_str} mm vs. long-term monthly mean {rf_mean_str} mm — trend {rf_trend}
  Temperature: latest {t_latest_str}°C vs. long-term mean {t_mean_str}°C (stable range for coastal Tunisia)
  Humidity  : {hum_str}% (long-term mean ~{cs.get('humidity_pct_mean', 0):.0f}%)
  Seasonality: rainfall CV = {cv_str} — {seasonality_text}
"""

    # --- Confidence explanation ---
    confidence = ic.confidence_score or 0
    confidence_basis = (
        "based on: multi-year climate record (15 yrs), dual-image temporal NDVI, "
        "full spectral index suite (NDVI/NDWI/NDBI), and regional ecosystem context. "
    )
    if confidence < 80:
        confidence_basis += "Score penalised for: limited water signal reducing flood confidence."

    # --- Safe formatting for None values in the main prompt ---
    ndvi_mean_str = f"{ic.ndvi_mean:.3f}" if ic.ndvi_mean is not None else "N/A"
    ndwi_mean_str = f"{ic.ndwi_mean:.3f}" if ic.ndwi_mean is not None else "N/A"
    ndbi_mean_str = f"{ic.ndbi_mean:.3f}" if ic.ndbi_mean is not None else "N/A"
    aridity_index_str = f"{ic.aridity_index:.3f}" if ic.aridity_index is not None else "N/A"
    ndvi_delta_str = f"{ic.ndvi_delta:+.3f}" if ic.ndvi_delta is not None else "N/A"
    
    # ═══════════════════════════════════════════════════════════════════════
    # THE PROMPT
    # ═══════════════════════════════════════════════════════════════════════
    prompt = f"""You are a senior environmental scientist specialising in arid and Mediterranean land systems.
You are writing a peer-review-quality geospatial intelligence report based on satellite imagery analysis and 15 years of climate data.

REGIONAL BASELINE KNOWLEDGE:
{regional_ctx}

═══ INPUT DATA ═══════════════════════════════════════════════════════════════

SPECTRAL INDICES (pre-interpreted):
  NDVI mean   : {ndvi_mean_str} — {ndvi_interp}
  NDWI mean   : {ndwi_mean_str} — {ndwi_interp}
  NDBI mean   : {ndbi_mean_str} — {ndbi_interp}

LAND COVER:
{lc_lines}
ARIDITY INDEX:
  Value : {aridity_index_str} — {aridity_interp}

TEMPORAL VEGETATION CHANGE (2010 → 2024):
  ΔNDVI : {ndvi_delta_str} — {delta_interp}

{climate_block}
RETRIEVED ENVIRONMENTAL CONTEXT:
{rag_context}

DETECTED SIGNALS:
{chr(10).join(f"  • {a}" for a in anomalies) if anomalies else "  • None"}
{contradiction_block}
CONFIDENCE SCORE: {confidence:.0f}% — {confidence_basis}

═══ WRITING INSTRUCTIONS ════════════════════════════════════════════════════

STRICT RULES — follow every one:
1. DO NOT restate raw numbers already provided above. Every number you mention must be accompanied by what it means ecologically or climatologically.
2. RESOLVE every contradiction listed above. Propose a physically plausible explanation.
3. REASON causally about anomalies. Do not just say "vegetation improved" — explain WHY it likely improved based on all available signals.
4. CONTEXTUALISE figures relative to regional baselines. Is 19°C mean temperature unusual here? Is 67% barren surprising? Is +0.188 ΔNDVI large or small for this ecosystem?
5. DO NOT include a section that simply answers "the user question". The entire report IS the answer.
6. DO NOT pad with phrases like "further monitoring is recommended" without specifying what to monitor, at what frequency, and why.
7. Use scientific language. Do not use phrases like "more favorable climate condition."

OUTPUT FORMAT — use exactly these sections:

## 1. Executive Summary
Two to three sentences. State the dominant land system state, the most significant finding, and the primary uncertainty.

## 2. Vegetation & Land Cover Assessment
Interpret vegetation density, spatial distribution, and land cover fractions in the context of the ecosystem type. Discuss what the NDVI value means for a Mediterranean/arid system specifically.

## 3. Temporal Vegetation Change (2010–2024)
Interpret the ΔNDVI change causally. What processes could explain it? Is the magnitude consistent with known greening trends in the Maghreb? What does it imply for land management?

## 4. Hydrological Assessment
Interpret the water fraction, NDWI signal, and any flooding/wetland signals together. Resolve any contradiction between them. Classify the likely water body type.

## 5. Climate–Vegetation Coupling
Analyse how the 15-year climate record (rainfall trend, temperature, humidity, seasonality) connects to the observed vegetation state and change. Address the rainfall seasonality issue explicitly.

## 6. Aridity & Drought Context
Interpret the aridity index in the context of both the long-term climate record and current conditions. Resolve any tension between aridity classification and current humidity/rainfall signals.

## 7. Key Findings & Ecological Implications
Three to five bullet points. Each must be a substantive scientific statement, not a data restatement.

## 8. Monitoring Recommendations
Two to three specific, actionable recommendations. Include: what index or variable to track, at what temporal frequency, and what threshold would trigger concern.

## 9. Confidence & Limitations
Explain the {confidence:.0f}% confidence score. State clearly what would raise it (e.g. in-situ validation, higher-resolution imagery) and what the main sources of uncertainty are.
"""

    return prompt


# ── Main report generation function ──────────────────────────────────────────

def generate_report(input_context, rag_context: str, anomalies: list[str]) -> str:
    """
    Generate the scientific report via Groq API.
    Returns the report as a formatted string.
    """
    client = get_groq_client()
    prompt = build_prompt(input_context, rag_context, anomalies)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior environmental scientist. "
                        "You write precise, evidence-based geospatial reports. "
                        "You never restate data without interpreting it. "
                        "You always resolve contradictions in the data rather than ignoring them. "
                        "You write for a technically literate audience."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,       # Lower = more consistent, less hallucination
            max_tokens=1800,       # Enough for a full structured report
        )

        report_text = response.choices[0].message.content.strip()

        # Wrap with standard header
        header = (
            "=" * 60 + "\n"
            "GEOSPATIAL INTELLIGENCE REPORT\n"
            "=" * 60 + "\n\n"
        )
        
        # Safe formatting for footer
        confidence_str = f"{input_context.confidence_score:.0f}" if input_context.confidence_score is not None else "N/A"
        region_str = input_context.region or 'Unknown'
        ecosystem_str = input_context.ecosystem or 'Unknown'
        
        footer = (
            "\n\n" + "=" * 60 + "\n"
            f"Confidence: {confidence_str}%  |  "
            f"Region: {region_str}  |  "
            f"Ecosystem: {ecosystem_str}\n"
            "=" * 60
        )

        return header + report_text + footer

    except Exception as e:
        return f"[Report generation failed: {e}]"
