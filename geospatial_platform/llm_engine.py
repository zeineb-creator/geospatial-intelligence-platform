import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geospatial_platform.validator import validate_inputs, build_reliability_report
from groq import Groq
from geospatial_platform.context import InputContext

MODEL_NAME = "llama-3.1-8b-instant"


def load_llm(hf_token: str = None):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to Streamlit secrets.")
    client = Groq(api_key=api_key)
    print(f"  Groq client initialized. Model: {MODEL_NAME}")
    return None, client


def compute_confidence(context: InputContext) -> float:
    score = 0.0

    # Base scores from available data
    if context.land_cover:                        score += 0.15
    if context.ndvi is not None:                  score += 0.15
    if context.ndwi is not None:                  score += 0.10
    if context.ndbi is not None:                  score += 0.08
    if context.csv_df is not None:                score += 0.12
    if context.vegetation_breakdown is not None:  score += 0.08
    if context.water_ratio is not None:           score += 0.05
    if context.aridity_index is not None:         score += 0.05

    # Penalties for missing critical data
    if context.ndvi_previous is None:             score -= 0.15  # no temporal
    if context.csv_df is None:                    score -= 0.10  # no climate
    if context.n_bands < 4:                       score -= 0.08  # RGB only

    # Multi-year climate bonus
    if context.csv_df is not None:
        if "year" in context.csv_df.columns:
            years = context.csv_df["year"].nunique()
            if years > 1:
                score += 0.10 * min(years / 5, 1.0)

    return round(max(0.10, min(score, 0.85)), 2)  # cap at 85%


def build_prompt(context: InputContext) -> list:
    system_message = (
        "You are a geospatial scientist specializing in remote sensing. "
        "STRICT RULES: "
        "1. Never infer vegetation CHANGE without temporal NDVI data — use 'current state' instead. "
        "2. Never claim flooding without NDWI confirmation. "
        "3. Never state causation — use 'may suggest', 'potentially indicates'. "
        "4. Always reference specific metric values in your conclusions. "
        "5. If a capability is marked Disabled in the reliability report — do NOT draw conclusions from it. "
        "6. Separate observations (what you see) from interpretations (what it might mean)."
    )

    flags            = validate_inputs(context)
    reliability_text = build_reliability_report(flags)

    lc = context.land_cover or {}
    vb = context.vegetation_breakdown or {}

    ndvi_mean  = round(float(context.ndvi.mean()), 3) if context.ndvi is not None else "N/A"
    ndwi_mean  = round(float(context.ndwi.mean()), 3) if context.ndwi is not None else "N/A"
    ndbi_mean  = round(float(context.ndbi.mean()), 3) if context.ndbi is not None else "N/A"
    water_ratio = f"{context.water_ratio*100:.1f}%" if context.water_ratio is not None else "N/A"
    aridity    = context.aridity_index if context.aridity_index else "N/A"

    # Vegetation change — only if real temporal data exists
    if context.ndvi_trend is not None:
        trend_text = f"ΔNDVI = {context.ndvi_trend:.3f} ({'decline' if context.ndvi_trend < 0 else 'improvement'})"
    else:
        trend_text = "NOT AVAILABLE — do not infer vegetation change"

    # Flood assessment
    if flags["flood_detectable"]:
        flood_text = f"Possible — water ratio {context.water_ratio*100:.1f}%, NDWI={ndwi_mean}"
    else:
        flood_text = "No clear flood evidence from spectral data"

    # Drought
    if flags["can_assess_drought"]:
        drought_text = f"Assessable — aridity index={aridity}"
    else:
        drought_text = "Limited assessment — missing rainfall or NDVI"

    anomalies_text = "\n".join(f"  - {a}" for a in (context.anomalies or [])) or "  None detected"
    confidence     = compute_confidence(context)
    context.confidence_score = confidence

    # Store flags in context for UI
    context.image_meta["validator_flags"] = flags
    context.image_meta["reliability_text"] = reliability_text

    user_message = f"""
{reliability_text}

SPECTRAL ANALYSIS:
  Land cover (mutually exclusive, adds to 100%):
    Water:      {lc.get('water', 0):.2f}%
    Vegetation: {lc.get('vegetation', 0):.2f}%
    Urban:      {lc.get('urban', 0):.2f}%
    Barren:     {lc.get('barren', 0):.2f}%

  Vegetation breakdown:
    Sparse   (NDVI 0.10–0.25): {vb.get('sparse_pct', 'N/A')}%
    Moderate (NDVI 0.25–0.45): {vb.get('moderate_pct', 'N/A')}%
    Dense    (NDVI > 0.45):    {vb.get('dense_pct', 'N/A')}%

  NDVI mean  : {ndvi_mean}
  NDWI mean  : {ndwi_mean}
  NDBI mean  : {ndbi_mean}
  NDVI trend : {trend_text}
  Water ratio: {water_ratio}
  Aridity    : {aridity}
  Flood      : {flood_text}
  Drought    : {drought_text}

CLIMATE DATA:
{context.retrieved_context or 'No climate data provided.'}

ANOMALIES:
{anomalies_text}

CONFIDENCE: {confidence*100:.1f}%

USER QUESTION: {context.user_question}

Write a structured scientific report:

1. OBSERVATIONS (what the data shows — no interpretation yet)
2. VEGETATION ASSESSMENT (use NDVI + breakdown; reference trend only if available)
3. HYDROLOGICAL ASSESSMENT (use NDWI + water ratio; distinguish water body vs flood)
4. CLIMATE CONTEXT (interpret temperature, rainfall, seasonality)
5. ENVIRONMENTAL INTERPRETATION (ecosystem type, stress factors — cautious language)
6. ANOMALY ANALYSIS (explain each anomaly; note if unsupported by data)
7. ANSWER TO USER QUESTION (direct, metric-referenced, cautious)
8. DATA RELIABILITY & LIMITATIONS (explicitly state what could not be assessed and why)
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message.strip()},
    ]


def generate_report(
    context: InputContext,
    tokenizer,
    llm,
    max_new_tokens: int = 1024,
) -> InputContext:
    print("=== LLM Reasoning Engine (Groq) ===")
    messages = build_prompt(context)

    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.2,
    )

    context.final_report = response.choices[0].message.content
    print("  Report generated via Groq.")
    print("=== LLM complete ===\n")
    return context
