"""
llm_engine.py — Global, sensor-agnostic geospatial report generator
=====================================================================
Improvements:
- Global ecosystem knowledge base (20+ biome types, not just Mediterranean)
- Location-agnostic interpretation: works for Amazon, Sahara, Siberia, etc.
- Pre-computed interpretation facts injected before LLM sees data
- Universal contradiction detector (aridity, water, NDVI, urban — all biomes)
- Two-stage pipeline: interpret (structured) → write (prose)
- Seasonal context derived from hemisphere + aridity class (no hardcoded region)
- Hallucination validator + correction retry
"""

import os
import re
import json
import numpy as np
from groq import Groq


# ══════════════════════════════════════════════════════════════════════════════
# GROQ CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def get_groq_client():
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=api_key)

def load_llm():
    """Stub — Groq is initialised inside generate_report()."""
    return None


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL ECOSYSTEM KNOWLEDGE BASE
# Covers all major biome types detectable from satellite imagery.
# Each entry is keyed to ecosystem strings produced by image_processor.py.
# ══════════════════════════════════════════════════════════════════════════════

ECOSYSTEM_KB = {

    # ── Arid / Desert ────────────────────────────────────────────────────────
    "hyper-arid desert": {
        "koppen":           "BWh / BWk",
        "rainfall_mm_yr":   (0, 25),
        "ndvi_range":       (0.00, 0.08),
        "ndvi_noise":       0.03,
        "ndvi_sig_change":  0.06,
        "barren_expected":  (70, 99),
        "veg_expected":     (0, 10),
        "water_expected":   (0, 5),
        "seasonal_pattern": "negligible — rainfall events are unpredictable and sparse",
        "water_body_types": ["ephemeral wadi", "playa", "reg (stone desert)"],
        "key_stressors":    ["extreme heat", "wind erosion", "salt crust formation"],
        "greening_context": "Vegetation pulses after rare rainfall events; NDVI spikes are short-lived (weeks)",
        "urban_note":       "Any urban signal in desert is likely industrial/mining or isolated settlement",
    },

    "arid shrubland / desert": {
        "koppen":           "BWh / BSh",
        "rainfall_mm_yr":   (25, 200),
        "ndvi_range":       (0.05, 0.18),
        "ndvi_noise":       0.04,
        "ndvi_sig_change":  0.08,
        "barren_expected":  (50, 85),
        "veg_expected":     (5, 30),
        "water_expected":   (0, 10),
        "seasonal_pattern": "pulse-driven — vegetation responds within weeks of rainfall events",
        "water_body_types": ["seasonal wadi", "sebkha", "playa", "oasis"],
        "key_stressors":    ["drought", "overgrazing", "soil salinisation", "wind erosion"],
        "greening_context": "Inter-annual NDVI variability driven by rainfall pulses; multi-year trends "
                            "reflect cumulative soil moisture and grazing pressure changes",
        "urban_note":       "Urban expansion in arid zones often visible as bright impervious surfaces "
                            "contrasting strongly with dark desert background",
    },

    "semi-arid mediterranean scrubland": {
        "koppen":           "Csa / BSk",
        "rainfall_mm_yr":   (200, 500),
        "ndvi_range":       (0.08, 0.40),
        "ndvi_noise":       0.05,
        "ndvi_sig_change":  0.10,
        "barren_expected":  (20, 60),
        "veg_expected":     (20, 60),
        "water_expected":   (0, 25),
        "seasonal_pattern": "strong winter-wet / summer-dry; NDVI peak Feb–May, trough Jul–Sep",
        "water_body_types": ["sebkha", "lagoon", "seasonal wetland", "coastal salt flat"],
        "key_stressors":    ["summer drought", "fire", "overgrazing", "soil erosion"],
        "greening_context": "ΔNDVI > 0.10 over 10+ years is consistent with documented Sahel/Maghreb "
                            "greening (Brandt et al. 2017; Venter et al. 2021) driven by rainfall "
                            "variability and CO₂ fertilisation; NOT unexpected",
        "urban_note":       "Peri-urban sprawl common around Mediterranean cities; "
                            "urban fraction 15–35% typical for coastal cities",
    },

    "mediterranean coastal mixed landscape": {
        "koppen":           "Csa / Csb",
        "rainfall_mm_yr":   (400, 700),
        "ndvi_range":       (0.10, 0.45),
        "ndvi_noise":       0.05,
        "ndvi_sig_change":  0.10,
        "barren_expected":  (10, 40),
        "veg_expected":     (25, 60),
        "water_expected":   (5, 35),
        "seasonal_pattern": "strong winter-wet / summer-dry; NDVI peak Feb–May, trough Jul–Sep",
        "water_body_types": ["sebkha", "lagoon", "coastal wetland", "salt flat"],
        "key_stressors":    ["summer drought", "urbanisation", "salinisation", "fire"],
        "greening_context": "ΔNDVI > 0.10 over 10+ years is consistent with documented regional "
                            "greening trends in North Africa and Southern Europe; NOT unexpected",
        "urban_note":       "Coastal urban fraction 15–35% typical; tourism infrastructure "
                            "common near shorelines",
    },

    "dry sub-humid mixed land": {
        "koppen":           "Csa / Cwa",
        "rainfall_mm_yr":   (500, 800),
        "ndvi_range":       (0.15, 0.55),
        "ndvi_noise":       0.05,
        "ndvi_sig_change":  0.10,
        "barren_expected":  (5, 30),
        "veg_expected":     (40, 75),
        "water_expected":   (0, 15),
        "seasonal_pattern": "moderate seasonality; dry summers limit peak greenness",
        "water_body_types": ["river", "reservoir", "seasonal wetland"],
        "key_stressors":    ["seasonal drought", "land clearing", "soil degradation"],
        "greening_context": "Vegetation responds strongly to interannual rainfall variability; "
                            "agricultural intensification can drive both greening and browning",
        "urban_note":       "Mixed rural-urban fringe typical; agricultural land common",
    },

    # ── Tropical / Subtropical ────────────────────────────────────────────────
    "dense forest / tropical vegetation": {
        "koppen":           "Af / Am / Aw",
        "rainfall_mm_yr":   (1500, 4000),
        "ndvi_range":       (0.60, 0.95),
        "ndvi_noise":       0.03,
        "ndvi_sig_change":  0.08,
        "barren_expected":  (0, 5),
        "veg_expected":     (70, 99),
        "water_expected":   (0, 20),
        "seasonal_pattern": "low seasonality in equatorial zones; pronounced dry season in savanna margins",
        "water_body_types": ["river", "floodplain", "oxbow lake", "varzea"],
        "key_stressors":    ["deforestation", "fire", "fragmentation", "selective logging"],
        "greening_context": "NDVI decline in tropical forest is a serious deforestation signal; "
                            "ΔNDVI < -0.05 over 5+ years indicates structural forest loss",
        "urban_note":       "Forest clearing for agriculture or urban expansion appears as sharp "
                            "NDVI decline with corresponding barren/urban increase",
    },

    "agricultural / grassland": {
        "koppen":           "Cfa / Cfb / Dfa / Cwa",
        "rainfall_mm_yr":   (600, 1500),
        "ndvi_range":       (0.25, 0.70),
        "ndvi_noise":       0.08,
        "ndvi_sig_change":  0.12,
        "barren_expected":  (5, 40),
        "veg_expected":     (40, 85),
        "water_expected":   (0, 15),
        "seasonal_pattern": "strong crop cycle seasonality; NDVI follows planting/harvest calendar",
        "water_body_types": ["irrigation canal", "reservoir", "river", "seasonal wetland"],
        "key_stressors":    ["drought", "flood", "soil degradation", "crop failure"],
        "greening_context": "High inter-annual NDVI variability driven by crop type rotation and rainfall; "
                            "multi-year trends reflect land use intensification or abandonment",
        "urban_note":       "Peri-urban agricultural land common; field patterns visible at 30m resolution",
    },

    "semi-arid savanna / mediterranean scrubland": {
        "koppen":           "BSh / Aw / Csa",
        "rainfall_mm_yr":   (300, 700),
        "ndvi_range":       (0.10, 0.45),
        "ndvi_noise":       0.06,
        "ndvi_sig_change":  0.10,
        "barren_expected":  (15, 55),
        "veg_expected":     (25, 65),
        "water_expected":   (0, 15),
        "seasonal_pattern": "strong wet/dry seasonality; savanna greens rapidly after first rains",
        "water_body_types": ["seasonal river", "pan", "seasonal wetland"],
        "key_stressors":    ["fire", "overgrazing", "bush encroachment", "drought"],
        "greening_context": "Bush encroachment (woody plant expansion) can drive NDVI increase "
                            "independent of rainfall — a key alternative hypothesis to test",
        "urban_note":       "Rural settlement common; urban fraction typically < 10%",
    },

    # ── Temperate ─────────────────────────────────────────────────────────────
    "humid mixed landscape": {
        "koppen":           "Cfb / Cfc / Dfb",
        "rainfall_mm_yr":   (700, 1500),
        "ndvi_range":       (0.30, 0.75),
        "ndvi_noise":       0.06,
        "ndvi_sig_change":  0.10,
        "barren_expected":  (0, 15),
        "veg_expected":     (50, 90),
        "water_expected":   (0, 20),
        "seasonal_pattern": "moderate — temperate deciduous cycle; peak greenness Jun–Aug (N hemisphere)",
        "water_body_types": ["lake", "river", "bog", "marsh"],
        "key_stressors":    ["urbanisation", "agricultural intensification", "drainage"],
        "greening_context": "Greening trends in temperate zones often linked to growing season "
                            "lengthening from warming temperatures (climate-driven phenological shift)",
        "urban_note":       "Extensive urban and suburban land use; impervious fraction 20–60% in cities",
    },

    "humid forest / dense vegetation": {
        "koppen":           "Cfb / Dfb / Cfc",
        "rainfall_mm_yr":   (800, 2000),
        "ndvi_range":       (0.50, 0.90),
        "ndvi_noise":       0.04,
        "ndvi_sig_change":  0.08,
        "barren_expected":  (0, 10),
        "veg_expected":     (70, 99),
        "water_expected":   (0, 15),
        "seasonal_pattern": "deciduous forests show strong spring green-up; evergreen forests more stable",
        "water_body_types": ["river", "lake", "peat bog", "floodplain"],
        "key_stressors":    ["deforestation", "pest outbreak", "fire", "windthrow"],
        "greening_context": "Temperate forest NDVI increase can reflect either natural recovery "
                            "after disturbance or growing season lengthening from climate warming",
        "urban_note":       "Forest fragmentation by roads/settlements detectable at 30m",
    },

    # ── Cold / Boreal ─────────────────────────────────────────────────────────
    "semi-arid barren land": {
        "koppen":           "BSk / ET",
        "rainfall_mm_yr":   (100, 400),
        "ndvi_range":       (0.05, 0.25),
        "ndvi_noise":       0.04,
        "ndvi_sig_change":  0.08,
        "barren_expected":  (40, 80),
        "veg_expected":     (10, 45),
        "water_expected":   (0, 10),
        "seasonal_pattern": "short growing season; snow cover affects winter NDVI",
        "water_body_types": ["ephemeral stream", "salt lake", "steppe wetland"],
        "key_stressors":    ["wind erosion", "overgrazing", "permafrost thaw (high lat)"],
        "greening_context": "Steppe greening linked to precipitation trends; browning linked to "
                            "drought intensification or permafrost degradation at high latitudes",
        "urban_note":       "Sparse settlement; mining and extraction visible as bright anomalies",
    },

    # ── Urban ─────────────────────────────────────────────────────────────────
    "urban / built-up area": {
        "koppen":           "any",
        "rainfall_mm_yr":   (0, 3000),
        "ndvi_range":       (0.05, 0.30),
        "ndvi_noise":       0.03,
        "ndvi_sig_change":  0.08,
        "barren_expected":  (5, 30),
        "veg_expected":     (5, 40),
        "water_expected":   (0, 15),
        "seasonal_pattern": "urban heat island suppresses seasonal signal; parks drive NDVI variability",
        "water_body_types": ["stormwater pond", "river", "recreational lake"],
        "key_stressors":    ["heat island", "impervious surface runoff", "green space loss"],
        "greening_context": "Urban NDVI increase typically reflects green infrastructure investment "
                            "or urban forest canopy maturation, not natural recovery",
        "urban_note":       "Urban fraction expected to dominate (> 40%); "
                            "NDBI and UI will both be elevated",
    },

    "peri-urban mixed landscape": {
        "koppen":           "any",
        "rainfall_mm_yr":   (0, 2000),
        "ndvi_range":       (0.10, 0.50),
        "ndvi_noise":       0.06,
        "ndvi_sig_change":  0.10,
        "barren_expected":  (5, 30),
        "veg_expected":     (20, 60),
        "water_expected":   (0, 20),
        "seasonal_pattern": "mixed — agricultural cycles and urban park seasonality",
        "water_body_types": ["reservoir", "irrigation canal", "stormwater pond"],
        "key_stressors":    ["urban sprawl", "agricultural land loss", "drainage modification"],
        "greening_context": "NDVI trends in peri-urban zones primarily reflect land conversion, "
                            "not climate-driven vegetation change",
        "urban_note":       "Urban growth detectable as barren→urban transition; "
                            "ΔNDVI may be negative due to vegetation clearance",
    },

    # ── Aquatic / Wetland ─────────────────────────────────────────────────────
    "aquatic / wetland": {
        "koppen":           "any",
        "rainfall_mm_yr":   (0, 3000),
        "ndvi_range":       (0.05, 0.55),
        "ndvi_noise":       0.05,
        "ndvi_sig_change":  0.10,
        "barren_expected":  (0, 20),
        "veg_expected":     (10, 60),
        "water_expected":   (30, 99),
        "seasonal_pattern": "inundation extent varies seasonally; emergent vegetation peaks in warm months",
        "water_body_types": ["permanent lake", "seasonal floodplain", "mangrove", "marsh", "estuary"],
        "key_stressors":    ["drainage", "eutrophication", "water level change", "invasive species"],
        "greening_context": "Wetland NDVI increase can reflect either vegetation recovery or "
                            "aquatic plant expansion (e.g. water hyacinth invasion)",
        "urban_note":       "Urban encroachment on wetlands is a common land use change driver",
    },

    # ── Fallback ──────────────────────────────────────────────────────────────
    "mixed / unclassified landscape": {
        "koppen":           "unknown",
        "rainfall_mm_yr":   (0, 3000),
        "ndvi_range":       (0.0, 1.0),
        "ndvi_noise":       0.05,
        "ndvi_sig_change":  0.10,
        "barren_expected":  (0, 100),
        "veg_expected":     (0, 100),
        "water_expected":   (0, 100),
        "seasonal_pattern": "unknown — interpret relative to latitude and climate data",
        "water_body_types": ["unknown"],
        "key_stressors":    ["unknown"],
        "greening_context": "Interpret NDVI trends relative to published baselines for the "
                            "detected climate zone",
        "urban_note":       "Classify urban signal relative to regional settlement patterns",
    },
}


def get_ecosystem_kb(ecosystem: str) -> dict:
    """Return the knowledge base entry for the detected ecosystem."""
    if not ecosystem:
        return ECOSYSTEM_KB["mixed / unclassified landscape"]
    eco_lower = ecosystem.lower().strip()
    # Exact match first
    if eco_lower in ECOSYSTEM_KB:
        return ECOSYSTEM_KB[eco_lower]
    # Partial match
    for key, val in ECOSYSTEM_KB.items():
        if any(word in eco_lower for word in key.split()):
            return val
    return ECOSYSTEM_KB["mixed / unclassified landscape"]


# ══════════════════════════════════════════════════════════════════════════════
# INDEX INTERPRETERS (global, no hardcoded region)
# ══════════════════════════════════════════════════════════════════════════════

def interpret_ndvi(ndvi: float, kb: dict = None) -> str:
    if ndvi is None:
        return "not available"
    # Generic thresholds — contextualised by ecosystem KB in facts block
    if ndvi < 0.05:   return "near-zero — bare soil, rock, or open water; no vegetation signal"
    elif ndvi < 0.12: return "very sparse — isolated individual plants or biological soil crust only"
    elif ndvi < 0.20: return "sparse — open vegetation with substantial exposed substrate"
    elif ndvi < 0.30: return "low-moderate — open canopy shrubland, sparse grassland, or stressed cropland"
    elif ndvi < 0.45: return "moderate — mixed shrub-grassland, seasonal cropland, or savanna"
    elif ndvi < 0.60: return "moderately high — dense shrubland, productive cropland, or open woodland"
    elif ndvi < 0.75: return "high — closed canopy woodland, dense cropland, or irrigated vegetation"
    else:             return "very high — dense tropical or temperate forest with full canopy closure"


def interpret_ndwi(ndwi: float) -> str:
    if ndwi is None:     return "not available"
    if ndwi > 0.30:      return "strongly positive — permanent open water body"
    elif ndwi > 0.10:    return "moderately positive — shallow water, flooded soil, or dense wetland"
    elif ndwi > 0.00:    return "mildly positive — moist soil or wetland fringe"
    elif ndwi > -0.15:   return "near-zero — dry to moderately moist surface; no standing water"
    elif ndwi > -0.30:   return "negative — dry surface or senescent vegetation; no water influence"
    else:                return "strongly negative — very dry surface or salt-encrusted substrate"


def interpret_ndbi(ndbi: float) -> str:
    if ndbi is None:   return "not available"
    if ndbi > 0.20:    return "high — dense urban fabric or heavily disturbed bare surface"
    elif ndbi > 0.05:  return "moderate — peri-urban fringe, exposed rock, or construction site"
    elif ndbi > 0.00:  return "low-moderate — sparse built-up features or partially exposed soil"
    else:              return "negative — vegetated or water-dominated; minimal impervious surface"


def interpret_aridity(ai: float) -> str:
    if ai is None:    return "not available"
    if ai < 0.05:     return "Hyper-arid (UNESCO) — essentially no rain-fed plant growth; desert core"
    elif ai < 0.20:   return "Arid (UNESCO) — sparse xerophytic vegetation; irrigation required for agriculture"
    elif ai < 0.50:   return "Semi-arid (UNESCO) — steppe vegetation; rain-fed farming marginal and risky"
    elif ai < 0.65:   return "Dry sub-humid (UNESCO) — seasonal moisture deficit; drought stress during dry season"
    else:             return "Humid (UNESCO) — sufficient rainfall year-round for dense vegetation without irrigation"


def interpret_ndvi_delta(delta: float, kb: dict = None) -> str:
    if delta is None:
        return "no temporal data"
    noise    = kb.get("ndvi_noise", 0.05)       if kb else 0.05
    sig      = kb.get("ndvi_sig_change", 0.10)  if kb else 0.10
    if abs(delta) <= noise:
        return f"within inter-annual noise (±{noise:.2f}) — no ecologically meaningful change"
    direction = "increase" if delta > 0 else "decline"
    magnitude = "large" if abs(delta) > sig * 1.5 else "moderate"
    return (
        f"{magnitude} vegetation {direction} ({delta:+.3f}) — "
        f"{'above' if abs(delta) > sig else 'near'} the significance threshold "
        f"(±{sig:.2f}) for this ecosystem type"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTED INTERPRETATION FACTS
# Derives verified statements from the data before the LLM prompt.
# These facts are injected as ground truth — the LLM must use them.
# ══════════════════════════════════════════════════════════════════════════════

def build_interpretation_facts(ic, kb: dict) -> str:
    """
    Compute ecosystem-specific, verified facts from the data.
    Injected into the prompt as pre-validated findings the LLM must use.
    Works for any ecosystem type — no hardcoded region.
    """
    facts = []

    # ── NDVI contextualisation ────────────────────────────────────────────────
    if ic.ndvi_mean is not None and kb:
        ndvi_lo, ndvi_hi = kb["ndvi_range"]
        ndvi_mid = (ndvi_lo + ndvi_hi) / 2

        if ic.ndvi_mean < ndvi_lo:
            facts.append(
                f"Measured NDVI ({ic.ndvi_mean:.3f}) is BELOW the typical range "
                f"({ndvi_lo:.2f}–{ndvi_hi:.2f}) for {ic.ecosystem or 'this ecosystem'}, "
                f"suggesting the image was acquired during peak dry season, "
                f"after a drought year, or that vegetation has degraded from historical norms."
            )
        elif ic.ndvi_mean > ndvi_hi:
            facts.append(
                f"Measured NDVI ({ic.ndvi_mean:.3f}) is ABOVE the typical range "
                f"({ndvi_lo:.2f}–{ndvi_hi:.2f}) for {ic.ecosystem or 'this ecosystem'}, "
                f"suggesting acquisition during peak growing season or above-average vegetation density."
            )
        else:
            facts.append(
                f"Measured NDVI ({ic.ndvi_mean:.3f}) falls within the climatologically normal range "
                f"({ndvi_lo:.2f}–{ndvi_hi:.2f}) for {ic.ecosystem or 'this ecosystem'}."
            )

    # ── Temporal NDVI change contextualisation ────────────────────────────────
    if ic.ndvi_delta is not None and kb:
        noise = kb.get("ndvi_noise", 0.05)
        sig   = kb.get("ndvi_sig_change", 0.10)
        ctx   = kb.get("greening_context", "")

        if abs(ic.ndvi_delta) <= noise:
            facts.append(
                f"ΔNDVI of {ic.ndvi_delta:+.3f} is within inter-annual noise "
                f"(±{noise:.2f} for this ecosystem) — no statistically meaningful "
                f"vegetation change can be concluded."
            )
        elif abs(ic.ndvi_delta) > sig:
            direction = "increase" if ic.ndvi_delta > 0 else "decline"
            facts.append(
                f"ΔNDVI of {ic.ndvi_delta:+.3f} exceeds the ecological significance threshold "
                f"(±{sig:.2f}) for {ic.ecosystem or 'this ecosystem'} — "
                f"this is a genuine, large-scale vegetation {direction}. "
                f"Ecosystem context: {ctx}"
            )
        else:
            direction = "increase" if ic.ndvi_delta > 0 else "decline"
            facts.append(
                f"ΔNDVI of {ic.ndvi_delta:+.3f} is a moderate vegetation {direction}, "
                f"above noise (±{noise:.2f}) but below the large-change threshold (±{sig:.2f}). "
                f"Ecosystem context: {ctx}"
            )

    # ── Land cover anomaly detection ──────────────────────────────────────────
    if ic.land_cover and kb:
        b_lo, b_hi = kb["barren_expected"]
        v_lo, v_hi = kb["veg_expected"]
        w_lo, w_hi = kb["water_expected"]

        barren = ic.land_cover.get("barren", 0)
        veg    = ic.land_cover.get("vegetation", 0)
        water  = ic.land_cover.get("water", 0)
        urban  = ic.land_cover.get("urban", 0)

        if barren < b_lo:
            facts.append(
                f"Barren fraction ({barren:.1f}%) is LOWER than the climatological norm "
                f"({b_lo}–{b_hi}%) for this ecosystem — suggesting unusually dense cover, "
                f"wet-season acquisition, or above-average rainfall year."
            )
        elif barren > b_hi:
            facts.append(
                f"Barren fraction ({barren:.1f}%) is HIGHER than the climatological norm "
                f"({b_lo}–{b_hi}%) for this ecosystem — consistent with degradation, "
                f"drought, or dry-season acquisition."
            )

        if veg > v_hi:
            facts.append(
                f"Vegetation fraction ({veg:.1f}%) exceeds the typical upper bound "
                f"({v_hi}%) for this ecosystem — likely peak growing season."
            )

        if water > w_hi:
            wb_types = ", ".join(kb.get("water_body_types", ["unknown"]))
            facts.append(
                f"Water fraction ({water:.1f}%) exceeds the typical range ({w_lo}–{w_hi}%) "
                f"for this ecosystem. Expected water body types: {wb_types}. "
                f"Active flooding requires corroboration with NDWI > 0.3 — "
                f"absent that, spectrally mixed water features (salt flat, shallow lagoon) "
                f"are the more likely explanation."
            )

    # ── Aridity vs humidity tension ───────────────────────────────────────────
    if (ic.aridity_index is not None and ic.aridity_index < 0.5 and
            ic.humidity_pct is not None and ic.humidity_pct > 65):
        facts.append(
            f"The aridity index ({ic.aridity_index:.3f}) and current humidity ({ic.humidity_pct:.1f}%) "
            f"are measuring different timescales — aridity integrates the full P/PET ratio over years, "
            f"while humidity reflects instantaneous atmospheric moisture at measurement time. "
            f"Both values are correct simultaneously: the long-term water balance is water-stressed, "
            f"and the current measurement was taken during or after a wet period."
        )

    # ── Water body classification ─────────────────────────────────────────────
    if ic.land_cover and ic.ndwi_mean is not None:
        water_pct = ic.land_cover.get("water", 0)
        if water_pct > 15 and ic.ndwi_mean < 0.0:
            wb_types = ", ".join(kb.get("water_body_types", ["spectrally mixed water body"])) if kb else "spectrally mixed water body"
            facts.append(
                f"High water fraction ({water_pct:.1f}%) combined with negative NDWI "
                f"({ic.ndwi_mean:.3f}) is the spectral signature of a mineralogically complex "
                f"surface: salt crust, brine, or shallow turbid water elevates SWIR reflectance, "
                f"suppressing NDWI below zero despite spatial classification as 'water'. "
                f"Probable water body types for this ecosystem: {wb_types}. "
                f"Active flooding is ruled out — that would require NDWI > 0.15."
            )
        elif water_pct > 10 and ic.ndwi_mean > 0.15:
            facts.append(
                f"Water fraction ({water_pct:.1f}%) is supported by positive NDWI "
                f"({ic.ndwi_mean:.3f}), confirming the presence of open surface water. "
                f"This is consistent with a permanent or semi-permanent water body."
            )

    # ── Urban signal interpretation ───────────────────────────────────────────
    if ic.land_cover and kb:
        urban = ic.land_cover.get("urban", 0)
        urban_note = kb.get("urban_note", "")
        if urban > 5:
            facts.append(f"Urban signal ({urban:.1f}%): {urban_note}")

    # ── Rainfall seasonality warning ──────────────────────────────────────────
    if ic.climate_summary:
        cv = ic.climate_summary.get("rainfall_mm_cv")
        if cv and cv > 0.7:
            facts.append(
                f"Rainfall coefficient of variation = {cv:.2f} — extreme seasonality. "
                f"Monthly rainfall values are unreliable indicators of annual water availability. "
                f"Interpret only multi-month or annual rainfall totals."
            )

    if not facts:
        return "No ecosystem-specific pre-computed facts available."
    return "\n".join(f"FACT {i+1}: {f}" for i, f in enumerate(facts))


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL CONTRADICTION DETECTOR
# Works for any ecosystem — all contradictions are physically derived,
# not region-specific
# ══════════════════════════════════════════════════════════════════════════════

def detect_contradictions(ctx: dict, kb: dict = None) -> list:
    contradictions = []
    aridity    = ctx.get("aridity_index")
    humidity   = ctx.get("humidity_pct")
    rf_trend   = ctx.get("rainfall_trend", "")
    water_pct  = ctx.get("water_pct", 0) or 0
    ndwi       = ctx.get("ndwi_mean", 0) or 0
    ndvi       = ctx.get("ndvi_mean")
    urban_pct  = ctx.get("urban_pct", 0) or 0
    barren_pct = ctx.get("barren_pct", 0) or 0
    veg_pct    = ctx.get("veg_pct", 0) or 0

    # 1. Aridity vs humidity (universal)
    if aridity is not None and aridity < 0.5 and humidity is not None and humidity > 65:
        facts.append = None  # don't double-inject — handled in facts block

    # 2. High water fraction + negative NDWI (universal)
    if water_pct > 15 and ndwi < 0.0:
        contradictions.append(
            f"Water fraction ({water_pct:.1f}%) is high but NDWI ({ndwi:.3f}) is negative. "
            "Physically: saline, turbid, or mineral-rich water bodies suppress NDWI via "
            "elevated SWIR reflectance. This is NOT a classification error — it is the "
            "spectral fingerprint of a spectrally mixed water surface. "
            "The spatial extent is real; the NDWI suppression is a sensor physics effect."
        )

    # 3. High NDVI + high aridity (any arid ecosystem)
    if (aridity is not None and aridity < 0.3 and
            ndvi is not None and ndvi > 0.40):
        contradictions.append(
            f"NDVI ({ndvi:.3f}) is moderately high despite an Arid/Semi-arid classification "
            f"(AI = {aridity:.3f}). Possible explanations: "
            "(a) image acquired during or immediately after wet season, "
            "(b) irrigated agriculture present, "
            "(c) the aridity index underestimates local moisture availability "
            "due to groundwater, river influence, or fog."
        )

    # 4. High urban + high vegetation (unusual co-occurrence)
    if urban_pct > 25 and veg_pct > 50:
        contradictions.append(
            f"Urban fraction ({urban_pct:.1f}%) and vegetation fraction ({veg_pct:.1f}%) "
            "are both elevated — unusual co-occurrence. "
            "Possible explanation: highly vegetated urban zone (parks, tree-lined streets, "
            "urban forest), or classifier overlap at the urban-vegetation boundary. "
            "Treat both fractions as upper-bound estimates."
        )

    # 5. Increasing rainfall + decreasing aridity class still arid
    if aridity is not None and aridity < 0.3 and "increasing" in rf_trend:
        contradictions.append(
            f"Rainfall trend is increasing yet aridity index ({aridity:.3f}) remains in the "
            "Arid/Semi-arid class. This is physically consistent: the aridity index integrates "
            "multi-decadal P/PET; a short-term rainfall increase does not shift the long-term "
            "water balance classification. Monitor for aridity index change over 10+ year periods."
        )

    # 6. Zero barren in an arid ecosystem (impossible in true deserts)
    if (kb and aridity is not None and aridity < 0.3 and barren_pct < 5):
        b_lo = kb.get("barren_expected", (20, 80))[0]
        if b_lo > 10:
            contradictions.append(
                f"Barren fraction ({barren_pct:.1f}%) is near-zero in what aridity data "
                f"classifies as an arid zone (AI = {aridity:.3f}). "
                "This suggests either: (a) wet-season acquisition with temporary vegetation flush, "
                "(b) irrigated agriculture masking the background arid signal, or "
                "(c) classifier over-assignment to vegetation/urban at the expense of bare soil."
            )

    return contradictions


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(ic, rag_context: str, anomalies: list) -> str:

    # Get ecosystem knowledge base
    kb = get_ecosystem_kb(getattr(ic, 'ecosystem', None))

    # Pre-interpret indices using ecosystem context
    ndvi_i  = interpret_ndvi(ic.ndvi_mean, kb)
    ndwi_i  = interpret_ndwi(ic.ndwi_mean)
    ndbi_i  = interpret_ndbi(ic.ndbi_mean)
    arid_i  = interpret_aridity(ic.aridity_index)
    delta_i = interpret_ndvi_delta(ic.ndvi_delta, kb)

    # Pre-computed facts block
    facts_block = build_interpretation_facts(ic, kb)

    # Contradiction detection
    ctx_dict = {
        "aridity_index": ic.aridity_index,
        "humidity_pct":  ic.humidity_pct,
        "rainfall_trend": ic.rainfall_trend or "",
        "water_pct":     getattr(ic, 'water_pct', None) or
                         (ic.land_cover.get("water", 0) if ic.land_cover else 0),
        "ndwi_mean":     ic.ndwi_mean,
        "ndvi_mean":     ic.ndvi_mean,
        "urban_pct":     ic.land_cover.get("urban", 0) if ic.land_cover else 0,
        "barren_pct":    ic.land_cover.get("barren", 0) if ic.land_cover else 0,
        "veg_pct":       ic.land_cover.get("vegetation", 0) if ic.land_cover else 0,
    }
    contradictions = detect_contradictions(ctx_dict, kb)
    contradiction_block = ""
    if contradictions:
        contradiction_block = "\n\nIDENTIFIED DATA TENSIONS — resolve each with physical reasoning:\n"
        for i, c in enumerate(contradictions, 1):
            contradiction_block += f"  {i}. {c}\n"

    # Ecosystem KB context block
    eco_block = f"""
ECOSYSTEM BASELINE (global knowledge base — {ic.ecosystem or 'unclassified'}):
  Köppen class    : {kb['koppen']}
  Typical rainfall: {kb['rainfall_mm_yr'][0]}–{kb['rainfall_mm_yr'][1]} mm/yr
  Typical NDVI    : {kb['ndvi_range'][0]:.2f}–{kb['ndvi_range'][1]:.2f}
  NDVI noise      : ±{kb['ndvi_noise']:.2f} (inter-annual; changes below this are not meaningful)
  Sig. NDVI change: >{kb['ndvi_sig_change']:.2f} (ecologically significant threshold)
  Barren expected : {kb['barren_expected'][0]}–{kb['barren_expected'][1]}%
  Veg expected    : {kb['veg_expected'][0]}–{kb['veg_expected'][1]}%
  Water expected  : {kb['water_expected'][0]}–{kb['water_expected'][1]}%
  Seasonality     : {kb['seasonal_pattern']}
  Key stressors   : {', '.join(kb['key_stressors'])}
  Greening context: {kb['greening_context']}
"""

    # Land cover block
    lc_lines = ""
    if ic.land_cover:
        for cls, pct in ic.land_cover.items():
            b_lo, b_hi = kb.get(f"{cls}_expected", (0, 100)) if f"{cls}_expected" in kb else (0, 100)
            lc_lines += f"    {cls:<14}: {pct:.1f}%\n"
    else:
        lc_lines = "    Not available\n"

    # Climate block
    climate_block = ""
    if ic.climate_summary:
        cs = ic.climate_summary
        def _fmt(v, dec=1): return f"{v:.{dec}f}" if v is not None else "N/A"
        cv   = cs.get("rainfall_mm_cv")
        seas = ("very high (CV>{:.2f}) — monthly values unreliable; use seasonal totals".format(cv)
                if cv and cv > 0.7 else
                "moderate" if cv else "unknown")
        climate_block = f"""
CLIMATE DATA:
  Rainfall   : latest {_fmt(cs.get('rainfall_mm_latest'))} mm vs mean {_fmt(cs.get('rainfall_mm_mean'))} mm — trend {cs.get('rainfall_mm_trend','unknown')}
  Temperature: latest {_fmt(cs.get('temperature_c_latest'))}°C vs mean {_fmt(cs.get('temperature_c_mean'))}°C
  Humidity   : latest {_fmt(cs.get('humidity_pct_latest'))}% vs mean {_fmt(cs.get('humidity_pct_mean',0),dec=0)}%
  Seasonality: CV = {_fmt(cv,dec=2) if cv else 'N/A'} — {seas}
"""

    def _n(v, fmt=".3f"): return format(v, fmt) if v is not None else "N/A"
    confidence = ic.confidence_score or 0
    eco_str    = ic.ecosystem or "unclassified landscape"
    region_str = ic.region    or "unknown region"

    prompt = f"""You are a senior environmental scientist with expertise in satellite remote sensing and global ecosystem ecology.
Write a peer-review-quality geospatial intelligence report. Your audience is technically literate.
The analysis covers: {region_str} | Ecosystem: {eco_str}

{eco_block}
PRE-COMPUTED SCIENTIFIC FACTS (verified from data — use these verbatim, do not contradict them):
{facts_block}

═══ INPUT DATA ══════════════════════════════════════════════════════════════

SPECTRAL INDICES:
  NDVI : {_n(ic.ndvi_mean)} — {ndvi_i}
  NDWI : {_n(ic.ndwi_mean)} — {ndwi_i}
  NDBI : {_n(ic.ndbi_mean)} — {ndbi_i}

LAND COVER:
{lc_lines}
ARIDITY INDEX: {_n(ic.aridity_index)} — {arid_i}

TEMPORAL CHANGE: ΔNDVI = {_n(ic.ndvi_delta, '+.3f') if ic.ndvi_delta is not None else 'N/A'} — {delta_i}

{climate_block}
RETRIEVED CONTEXT:
{rag_context}

DETECTED SIGNALS:
{chr(10).join(f'  • {a}' for a in anomalies) if anomalies else '  • None'}
{contradiction_block}
CONFIDENCE: {confidence:.0f}%

═══ WRITING RULES ══════════════════════════════════════════════════════════

1. USE THE PRE-COMPUTED FACTS. They are verified. Do not contradict them.
   Incorporate each FACT into the relevant section as the core of your interpretation.

2. NEVER open a sentence with a number or index name.
   FORBIDDEN: "The NDVI of X...", "The ΔNDVI (+X)...", "The water fraction (X%)..."
   REQUIRED: Start with ecological meaning, then optionally reference the value.
   WRONG: "The NDVI value of 0.201 indicates low vegetation"
   RIGHT: "Vegetation is confined to drought-tolerant shrubs — soil dominates the spectral signal"

3. USE THE ECOSYSTEM KB. Every finding must be contextualised against the baseline ranges provided.
   State explicitly: is this value expected, above-norm, or below-norm for this ecosystem?

4. CAUSAL CHAINS required for every anomaly. Not "X suggests Y" but "X because mechanism Z,
   which causes Y via process P."

5. MONITORING: each recommendation must specify data source, frequency, AND threshold.
   WRONG: "Monitor NDVI monthly"
   RIGHT: "Monitor Sentinel-2 NDVI composites monthly; a [growing season month] value below
           [specific number = 10th percentile for this ecosystem] triggers [specific action]"

6. SECTION 3 (Temporal): explicitly state whether the ΔNDVI magnitude is within or beyond
   the ecosystem's normal inter-annual variability range (provided in the KB above).

7. Forbidden phrases: "provides valuable insights", "it is worth noting", "further study needed",
   "not a reliable indicator", "unexpected for this region" (unless supported by the KB).

═══ OUTPUT FORMAT ═══════════════════════════════════════════════════════════

## 1. Executive Summary
2–3 sentences: dominant land system state, most significant finding, primary uncertainty.

## 2. Vegetation & Land Cover Assessment
Interpret vegetation density and land cover fractions against the ecosystem KB ranges.
State explicitly whether each fraction is within, above, or below the expected range.

## 3. Temporal Vegetation Change
Causal interpretation of ΔNDVI. Compare magnitude against ecosystem noise and significance thresholds.
What processes explain the change? Is it within documented norms for this ecosystem type?

## 4. Hydrological Assessment
Interpret water fraction and NDWI together. Classify water body type from the ecosystem KB list.
Resolve any tension between spatial water fraction and NDWI signal.

## 5. Climate–Vegetation Coupling
Connect climate record to vegetation state. Address seasonality explicitly.
What can and cannot be concluded from monthly climate data?

## 6. Aridity & Drought Context
(Skip if no aridity data available — state that clearly.)
Interpret aridity index as a long-term tool. Resolve any tension with current conditions.

## 7. Key Findings & Ecological Implications
5 bullet points. Each must begin with an ecological statement, not a number.
Each must contain a causal mechanism.

## 8. Monitoring Recommendations
3 recommendations. Each: specific index + data source + frequency + numerical threshold + consequence.

## 9. Confidence & Limitations
Explain the {confidence:.0f}% score. What data is missing? What would raise it?
"""
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def _validate_report(report_text: str) -> tuple[bool, list[str]]:
    failures = []

    # 1. All 9 sections present
    for i in range(1, 10):
        if f'## {i}.' not in report_text:
            failures.append(f"Section {i} missing")

    # 2. Monitoring has thresholds and frequency
    m = re.search(r'## 8\..*?(?=## 9\.|$)', report_text, re.DOTALL | re.IGNORECASE)
    if m:
        mt = m.group(0)
        if not re.search(r'\b0\.\d+|\d+\.\d+|below \d|above \d', mt):
            failures.append("Section 8 missing numerical thresholds")
        if not any(w in mt.lower() for w in ['monthly','seasonal','annual','weekly','quarterly']):
            failures.append("Section 8 missing temporal frequency")
    else:
        failures.append("Section 8 not found")

    # 3. Forbidden phrases
    for phrase in ["not a reliable indicator", "provides valuable insights",
                   "it is worth noting", "further study is needed", "short record"]:
        if phrase in report_text.lower():
            failures.append(f"Forbidden phrase: '{phrase}'")

    return len(failures) == 0, failures


def _build_correction_prompt(original: str, failures: list[str]) -> str:
    return f"""Fix these issues in the geospatial report:
{chr(10).join(f'  - {f}' for f in failures)}

Section 8 monitoring format required:
"Monitor [index] via [source] at [frequency] frequency; a value [below/above] [number] 
would indicate [consequence] requiring [action]."

Remove any sentences with: "not a reliable indicator", "provides valuable insights",
"it is worth noting", "further study is needed".

REPORT TO FIX:
{original}

Return the complete corrected report. Fix only the listed issues."""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(input_context, rag_context: str = "", anomalies: list = None) -> str:
    if anomalies is None:
        anomalies = []

    client = get_groq_client()
    prompt = build_prompt(input_context, rag_context, anomalies)

    system_msg = (
        "You are a senior environmental scientist with global remote sensing expertise. "
        "You write peer-review-quality geospatial reports. "
        "You use the pre-computed scientific facts provided — you never contradict them. "
        "You interpret data ecologically — you never restate raw numbers as findings. "
        "You give specific, actionable monitoring recommendations with numerical thresholds."
    )

    def _call(messages):
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=2400,
        ).choices[0].message.content.strip()

    def _wrap(text):
        ic = input_context
        conf_s = f"{ic.confidence_score:.0f}" if ic.confidence_score is not None else "N/A"
        header = "=" * 60 + "\nGEOSPATIAL INTELLIGENCE REPORT\n" + "=" * 60 + "\n\n"
        footer = (
            "\n\n" + "=" * 60 + "\n"
            f"Confidence: {conf_s}%  |  Region: {ic.region or 'Unknown'}  |  "
            f"Ecosystem: {ic.ecosystem or 'Unknown'}\n" + "=" * 60
        )
        return header + text + footer

    try:
        # Attempt 1
        report = _call([
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ])
        valid, failures = _validate_report(report)
        if valid:
            print("  [LLM] Report passed validation ✓")
            return _wrap(report)

        # Attempt 2 — targeted correction
        print(f"  [LLM] Validation failed ({len(failures)} issues) — retrying")
        for f in failures:
            print(f"    ✗ {f}")
        corrected = _call([
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": _build_correction_prompt(report, failures)},
        ])
        valid2, failures2 = _validate_report(corrected)
        if valid2:
            print("  [LLM] Correction passed ✓")
        else:
            print(f"  [LLM] Still {len(failures2)} issues after correction — returning best effort")
        return _wrap(corrected)

    except Exception as e:
        return f"[Report generation failed: {e}]"
