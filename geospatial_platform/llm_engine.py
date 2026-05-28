import os
from groq import Groq
from typing import Optional
import numpy as np


# ── Groq client ───────────────────────────────────────────────────────────────

def get_groq_client():
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=api_key)


# ── Index interpretation helpers ──────────────────────────────────────────────

def interpret_ndvi(ndvi: float) -> str:
    if ndvi is None: return "not available"
    if ndvi < 0.1:   return "near-zero — bare soil or rock, no meaningful vegetation signal"
    elif ndvi < 0.2: return "very low — sparse or stressed xerophytic vegetation typical of arid zones; soil dominates reflectance"
    elif ndvi < 0.35: return "low-moderate — open shrubland or degraded vegetation; seasonal greening likely limited to cooler months"
    elif ndvi < 0.5: return "moderate — mixed shrub-grassland or seasonally active vegetation; productive during wet season"
    elif ndvi < 0.65: return "moderately high — dense shrubland or productive cropland with active canopy"
    else:             return "high — dense healthy canopy consistent with forest or irrigated agriculture"


def interpret_ndwi(ndwi: float) -> str:
    if ndwi is None:  return "not available"
    if ndwi > 0.3:    return "strongly positive — open water body clearly present; likely permanent surface water"
    elif ndwi > 0.0:  return "mildly positive — wet soils or shallow water influence; possible wetland fringe"
    elif ndwi > -0.15: return "near-zero — dry to moist surface with no open water signal"
    else:              return "negative — dry soil or senescent vegetation with no surface moisture signal"


def interpret_ndbi(ndbi: float) -> str:
    if ndbi is None:  return "not available"
    if ndbi > 0.2:    return "high — significant impervious surface; dense urban fabric"
    elif ndbi > 0.0:  return "low-moderate — sparse built-up features, peri-urban fringe, or exposed rock"
    else:              return "negative — vegetated or water-dominated surface; minimal built-up signal"


def interpret_aridity(ai: float) -> str:
    if ai is None:    return "not available"
    if ai < 0.05:     return "Hyper-arid — essentially no rain-fed vegetation possible; desert core"
    elif ai < 0.2:    return "Arid — sparse xerophytic vegetation; all agriculture requires irrigation"
    elif ai < 0.5:    return "Semi-arid — steppe ecosystems; rain-fed agriculture marginal and drought-prone"
    elif ai < 0.65:   return "Dry sub-humid — Mediterranean-type; vegetation under significant summer drought stress"
    else:              return "Humid — sufficient rainfall for dense vegetation without irrigation"


def interpret_ndvi_delta(delta: float) -> str:
    if delta is None: return "no temporal data"
    if delta > 0.15:
        return (f"large positive shift (+{delta:.3f}) — well above inter-annual noise (±0.02–0.05 typical); "
                "consistent with multi-year land recovery, reforestation, or sustained wet-period greening")
    elif delta > 0.05:
        return (f"moderate positive shift (+{delta:.3f}) — above noise threshold; "
                "gradual greening consistent with reduced grazing pressure or improving rainfall")
    elif delta > -0.05:
        return f"stable (±{delta:.3f}) — within inter-annual noise; no ecologically meaningful change"
    elif delta > -0.15:
        return (f"moderate decline ({delta:.3f}) — vegetation loss beyond noise; "
                "possible drought stress, overgrazing, or land clearing")
    else:
        return (f"large decline ({delta:.3f}) — severe vegetation loss; "
                "consistent with desertification, fire, or major land use change")


def get_regional_context(ecosystem: str, region: str) -> str:
    eco_lower = ecosystem.lower() if ecosystem else ""
    if "mediterranean" in eco_lower:
        base = (
            "Mediterranean ecosystems follow a strongly seasonal pattern (Köppen Csa/Csb): "
            "mild wet winters (Oct–Mar, 400–600 mm in coastal Tunisia) and hot dry summers. "
            "NDVI peaks Feb–May and drops sharply Jun–Sep as vegetation senesces. "
            "Aridity index below 0.5 indicates significant dry-season stress even when annual totals appear adequate — "
            "the index reflects the long-term water balance, NOT instantaneous conditions. "
            "Water fractions above 20% in a coastal Tunisian scene are most consistent with "
            "a sebkha (salt flat), lagoon, or shallow wetland — not active flooding, "
            "since flooding would require an extreme event inconsistent with the stable NDWI signal. "
            "Inter-annual NDVI noise in this biome is ±0.02–0.05; changes above 0.10 are ecologically significant."
        )
    elif "arid" in eco_lower or "semi-arid" in eco_lower:
        base = (
            "Arid and semi-arid zones have highly variable interannual rainfall. "
            "Vegetation responds rapidly to rainfall pulses; NDVI can spike after one wet season and recede quickly. "
            "Barren fractions above 60% are climatically normal. "
            "Inter-annual NDVI noise is ±0.03–0.06."
        )
    else:
        base = "No specific ecosystem baseline. Interpret indices relative to global published norms."

    if region:
        base += f" Analysis region: {region}."
    return base


# ── Contradiction detector ────────────────────────────────────────────────────

def detect_contradictions(context: dict) -> list:
    contradictions = []
    aridity       = context.get("aridity_index")
    humidity      = context.get("humidity_pct")
    rf_trend      = context.get("rainfall_trend", "")
    water_pct     = context.get("water_pct", 0) or 0
    ndwi          = context.get("ndwi_mean", 0) or 0

    if aridity is not None and aridity < 0.2 and humidity is not None and humidity > 65:
        contradictions.append(
            f"Aridity index ({aridity:.3f}) is a long-term climatological measure classifying the area as Arid. "
            f"Current humidity ({humidity:.1f}%) is above the long-term mean. "
            "This is NOT a contradiction in the index's reliability — it reflects seasonal timing: "
            "the measurement was likely taken during or after the wet season. "
            "The aridity index correctly characterises the multi-decadal water balance."
        )

    if water_pct > 20 and ndwi < 0:
        contradictions.append(
            f"Land cover classification shows {water_pct:.1f}% water fraction, "
            f"yet NDWI is {ndwi:.3f} (negative). "
            "This is consistent with a spectrally mixed coastal feature — sebkha or salt flat — "
            "where high mineral salt content elevates SWIR reflectance, suppressing NDWI below zero "
            "even where the surface is classified as 'water' by the threshold-based classifier. "
            "Active flooding is ruled out."
        )

    if aridity is not None and aridity < 0.2 and "increasing" in rf_trend:
        contradictions.append(
            f"Rainfall trend is increasing yet the aridity index ({aridity:.3f}) remains in the Arid class. "
            "This indicates the recent wet trend is short-term and insufficient to shift the multi-decadal "
            "Precipitation/PET ratio into the semi-arid range — consistent with observed sub-decadal "
            "rainfall variability in the Maghreb without structural climate regime change."
        )
    return contradictions


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(ic, rag_context: str, anomalies: list) -> str:

    # Pre-interpret indices
    ndvi_i   = interpret_ndvi(ic.ndvi_mean)
    ndwi_i   = interpret_ndwi(ic.ndwi_mean)
    ndbi_i   = interpret_ndbi(ic.ndbi_mean)
    arid_i   = interpret_aridity(ic.aridity_index)
    delta_i  = interpret_ndvi_delta(ic.ndvi_delta)

    # Contradictions
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
        contradiction_block = "\n\nIDENTIFIED DATA TENSIONS — you MUST address each one scientifically:\n"
        for i, c in enumerate(contradictions, 1):
            contradiction_block += f"  {i}. {c}\n"

    # Regional context
    regional_ctx = get_regional_context(ic.ecosystem, ic.region)

    # Land cover block
    lc_lines = ""
    if ic.land_cover:
        for cls, pct in ic.land_cover.items():
            lc_lines += f"    {cls:<14}: {pct:.1f}%\n"
    else:
        lc_lines = "    Not available\n"

    # Climate block
    climate_block = ""
    if ic.climate_summary:
        cs = ic.climate_summary
        def _fmt(v, dec=1): return f"{v:.{dec}f}" if v is not None else "N/A"
        rf_l  = _fmt(cs.get("rainfall_mm_latest"))
        rf_m  = _fmt(cs.get("rainfall_mm_mean"))
        rf_tr = cs.get("rainfall_mm_trend", "unknown")
        t_l   = _fmt(cs.get("temperature_c_latest"))
        t_m   = _fmt(cs.get("temperature_c_mean"))
        h_l   = _fmt(cs.get("humidity_pct_latest"))
        h_m   = _fmt(cs.get("humidity_pct_mean", 0), dec=0)
        cv    = cs.get("rainfall_cv")
        cv_s  = _fmt(cv, dec=2) if cv is not None else "N/A"
        seas  = ("very high — monthly values are unreliable indicators of annual conditions"
                 if cv and cv > 0.7 else "moderate")
        climate_block = f"""
CLIMATE DATA (15-year record):
  Rainfall   : latest {rf_l} mm vs long-term monthly mean {rf_m} mm — trend {rf_tr}
  Temperature: latest {t_l}°C vs long-term mean {t_m}°C
  Humidity   : latest {h_l}% vs long-term mean {h_m}%
  Seasonality: rainfall CV = {cv_s} — {seas}
"""

    # Safe number formatting
    def _n(v, fmt=".3f"): return format(v, fmt) if v is not None else "N/A"

    confidence     = ic.confidence_score or 0
    conf_basis     = ("multi-year climate record, dual-image temporal NDVI, full spectral suite, regional context")
    if confidence < 80:
        conf_basis += "; score reduced: limited water-body validation data"

    prompt = f"""You are a senior environmental scientist specialising in arid and Mediterranean land systems.
Write a peer-review-quality geospatial intelligence report. Your audience is technically literate.

REGIONAL BASELINE KNOWLEDGE:
{regional_ctx}

═══ PROCESSED INPUT DATA ════════════════════════════════════════════════════

SPECTRAL INDICES (already interpreted — do not re-describe the numbers):
  NDVI mean : {_n(ic.ndvi_mean)} — {ndvi_i}
  NDWI mean : {_n(ic.ndwi_mean)} — {ndwi_i}
  NDBI mean : {_n(ic.ndbi_mean)} — {ndbi_i}

LAND COVER:
{lc_lines}
ARIDITY INDEX:
  {_n(ic.aridity_index)} — {arid_i}

TEMPORAL VEGETATION CHANGE:
  ΔNDVI : {_n(ic.ndvi_delta, '+.3f') if ic.ndvi_delta is not None else 'N/A'} — {delta_i}

{climate_block}
RETRIEVED CONTEXT:
{rag_context}

DETECTED SIGNALS:
{chr(10).join(f'  • {a}' for a in anomalies) if anomalies else '  • None'}
{contradiction_block}
CONFIDENCE: {confidence:.0f}% — {conf_basis}

═══ WRITING RULES (violations = failed peer review) ════════════════════════

1. NEVER open a sentence with a number or index name. The data section already contains all numbers.
   Your job is to write ONLY what the numbers mean — cause, consequence, ecological context.
   FORBIDDEN OPENINGS: "The NDVI value of X", "The ΔNDVI of +X", "The water fraction (X%)",
                       "The aridity index (X)", "The urban fraction (X%)"
   REQUIRED: Start every interpretive sentence with the ecological or physical meaning.
   WRONG: "The NDVI value of 0.201 indicates low-moderate vegetation cover"
   RIGHT: "Vegetation is confined to drought-tolerant shrubs persisting through summer senescence —
           the spectral signal is dominated by exposed soil rather than canopy"
   WRONG: "The large positive shift in NDVI (+0.192) indicates substantial improvement"
   RIGHT: "Shrub canopy density nearly doubled over the 14-year record, a rate consistent
           with documented post-drought recovery trajectories in the Maghreb"
   WRONG in Section 7 bullets: "The large positive shift in NDVI (+0.192) indicates..."
   RIGHT in Section 7 bullets: "Shrub recovery accelerated between 2010 and 2024 at a rate..."
   Apply this rule to EVERY sentence in EVERY section including bullet points.

2. RESOLVE data tensions using physical mechanisms — never dismiss an index as "unreliable".
   The aridity index is a valid long-term climatological measure. Seasonal humidity above the mean
   does not invalidate it — explain WHY both can be true simultaneously.

3. CAUSAL chains are required for all anomalies.
   WRONG: "vegetation improved, likely driven by increased rainfall"
   RIGHT: "Consecutive wet seasons accumulate soil moisture beyond single-season capacity,
           enabling shrub root systems to access deeper water reserves and sustain growth
           into the dry season — a documented mechanism in post-drought Maghreb recovery"
   FACTUAL REQUIREMENT: A ΔNDVI of +0.15 or greater over 10+ years in the Maghreb is
   CONSISTENT WITH — not unexpected for — documented regional greening trends driven by
   increased rainfall variability and CO₂ fertilisation effects. Do NOT describe this as
   "unexpected" or "surprising". The correct framing is: large but within the observed
   range of Sahel/Maghreb greening documented in the peer-reviewed literature.

4. CONTEXTUALISE using the regional baseline provided. State explicitly whether each finding
   is surprising, expected, or anomalous for this ecosystem type.

5. Each monitoring recommendation MUST include:
   (a) exact variable/index and data source (e.g. "Sentinel-2 NDVI composite")
   (b) temporal frequency (monthly / seasonal / annual)
   (c) specific numerical threshold that would trigger management concern
   WRONG: "Track NDVI at monthly frequency to monitor vegetation"
   RIGHT: "Monitor Sentinel-2 NDVI monthly composites; an April–May value below 0.12
           (below the 10th percentile for this ecosystem) would indicate drought stress
           requiring assessment of irrigation or reforestation intervention"

6. Forbidden phrases: "more favorable", "not a reliable indicator", "further study needed",
   "provides valuable insights", "it is worth noting", "it is important to".

═══ OUTPUT FORMAT ════════════════════════════════════════════════════════════

## 1. Executive Summary
2–3 sentences: dominant land system state, most significant finding, primary uncertainty.

## 2. Vegetation & Land Cover Assessment
Interpret vegetation density and land cover fractions ecologically.
What does this NDVI value specifically mean for a {ic.ecosystem or 'Mediterranean'} system?
Is the barren fraction climatically expected or anomalous?

## 3. Temporal Vegetation Change
Causal interpretation of ΔNDVI. Is this magnitude expected for the Maghreb?
What land management implications follow?

## 4. Hydrological Assessment
Interpret water fraction, NDWI, and water body type together.
Resolve any tension. Rule flooding in or out with reasoning.

## 5. Climate–Vegetation Coupling
How does the 15-year record connect to observed vegetation state and change?
Address rainfall seasonality explicitly — what can and cannot be concluded from monthly data?

## 6. Aridity & Drought Context
Interpret aridity index as a long-term climatological tool.
Explain how seasonal humidity and the aridity classification can both be correct simultaneously.

## 7. Key Findings & Ecological Implications
5 bullet points. Each must be a substantive scientific statement with a causal claim —
not a data restatement, not a hedged observation.

## 8. Monitoring Recommendations
3 recommendations. Each MUST include variable, data source, frequency, and threshold with consequence.

## 9. Confidence & Limitations
Explain the {confidence:.0f}% score. What would raise it? What are the main uncertainty sources?
"""
    return prompt



# ── Output validator ──────────────────────────────────────────────────────────

def _validate_report(report_text: str) -> tuple[bool, list[str]]:
    """
    Check the LLM output meets minimum scientific quality standards.
    Returns (is_valid, list_of_failures).
    """
    failures = []
    text = report_text.lower()

    # 1. Monitoring section must contain numerical thresholds
    import re
    monitoring_match = re.search(
        r'## 8\..*?(?=## 9\.|$)', report_text, re.DOTALL | re.IGNORECASE
    )
    if monitoring_match:
        monitoring_text = monitoring_match.group(0)
        # Must have at least one number that looks like a threshold
        has_threshold = bool(re.search(r'\b0\.\d+|\d+\.\d+%|below \d|above \d|threshold', monitoring_text))
        has_frequency = any(w in monitoring_text.lower() for w in
                           ['monthly', 'seasonal', 'annual', 'weekly', 'quarterly'])
        if not has_threshold:
            failures.append("Section 8 (Monitoring) is missing numerical thresholds")
        if not has_frequency:
            failures.append("Section 8 (Monitoring) is missing temporal frequency")
    else:
        failures.append("Section 8 (Monitoring) not found in output")

    # 2. Must have all 9 sections
    for i in range(1, 10):
        if f'## {i}.' not in report_text and f'## {i} .' not in report_text:
            failures.append(f"Section {i} missing from report")

    # 3. Must not contain the "short record" hallucination or "not a reliable"
    bad_phrases = [
        "short climate record",
        "short record",
        "not a reliable indicator",
        "provides valuable insights",
        "it is worth noting",
        "further study is needed",
    ]
    for phrase in bad_phrases:
        if phrase in text:
            failures.append(f"Forbidden phrase detected: \"{phrase}\"")

    return len(failures) == 0, failures


def _build_correction_prompt(original_report: str, failures: list[str]) -> str:
    """Build a targeted correction prompt for failed sections only."""
    failure_str = "\n".join(f"  - {f}" for f in failures)
    return f"""The following geospatial report has quality issues that must be fixed:

ISSUES TO FIX:
{failure_str}

SPECIFIC REQUIREMENTS FOR SECTION 8 (Monitoring Recommendations):
Each of the 3 recommendations MUST follow this exact format:
"Monitor [specific index] via [data source] at [monthly/seasonal/annual] frequency; \
a value [below/above] [specific number] would indicate [specific ecological consequence] \
requiring [specific action]."

Example of acceptable recommendation:
"Monitor Sentinel-2 NDVI monthly composites (April–May growing season window); \
a mean value below 0.10 — below the 5th percentile for Mediterranean coastal shrubland — \
would indicate severe drought stress requiring assessment of emergency irrigation or \
reforestation programme suspension."

REMOVE any sentences containing these forbidden phrases:
- "short climate record" or "short record" (the record is 15 years — not short)
- "not a reliable indicator"
- "provides valuable insights"
- "it is worth noting"

ORIGINAL REPORT TO FIX:
{original_report}

Return the complete corrected report with all 9 sections. Fix ONLY the issues listed above.
Keep all other content unchanged."""


# ── Main generation function ──────────────────────────────────────────────────

def generate_report(input_context, rag_context: str, anomalies: list) -> str:
    client = get_groq_client()
    prompt = build_prompt(input_context, rag_context, anomalies)

    system_msg = (
        "You are a senior environmental scientist writing peer-review-quality reports. "
        "You interpret data — you never restate it. "
        "You explain physical mechanisms — you never use vague causal language. "
        "You give specific, actionable recommendations with numerical thresholds. "
        "You never dismiss a validated index as 'unreliable'."
    )

    def _call_llm(messages):
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.25,
            max_tokens=2400,
        ).choices[0].message.content.strip()

    def _wrap(text):
        header = "=" * 60 + "\nGEOSPATIAL INTELLIGENCE REPORT\n" + "=" * 60 + "\n\n"
        conf_s = f"{input_context.confidence_score:.0f}" if input_context.confidence_score is not None else "N/A"
        footer = (
            "\n\n" + "=" * 60 + "\n"
            f"Confidence: {conf_s}%  |  "
            f"Region: {input_context.region or 'Unknown'}  |  "
            f"Ecosystem: {input_context.ecosystem or 'Unknown'}\n"
            + "=" * 60
        )
        return header + text + footer

    try:
        # ── Attempt 1: standard generation ───────────────────────────────────
        report_text = _call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ])

        is_valid, failures = _validate_report(report_text)

        if is_valid:
            print(f"  [LLM] Report passed validation on first attempt.")
            return _wrap(report_text)

        # ── Attempt 2: targeted correction ────────────────────────────────────
        print(f"  [LLM] Validation failed ({len(failures)} issues). Retrying with correction prompt...")
        for f in failures:
            print(f"    ✗ {f}")

        correction_prompt = _build_correction_prompt(report_text, failures)
        report_text_v2 = _call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": correction_prompt},
        ])

        is_valid_v2, failures_v2 = _validate_report(report_text_v2)
        if is_valid_v2:
            print(f"  [LLM] Report passed validation after correction.")
        else:
            print(f"  [LLM] Report still has issues after correction ({len(failures_v2)} remaining):")
            for f in failures_v2:
                print(f"    ✗ {f}")
            # Return corrected version anyway — it will be better than the original

        return _wrap(report_text_v2)

    except Exception as e:
        return f"[Report generation failed: {e}]"
