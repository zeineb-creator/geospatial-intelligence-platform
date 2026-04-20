import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    if context.land_cover:                        score += 0.20
    if context.ndvi is not None:                  score += 0.20
    if context.ndwi is not None:                  score += 0.15
    if context.ndbi is not None:                  score += 0.10
    if context.csv_df is not None:                score += 0.15
    if context.vegetation_breakdown is not None:  score += 0.10
    if context.water_ratio is not None:           score += 0.05
    if context.aridity_index is not None:         score += 0.05
    # Penalize missing temporal data
    if context.ndvi_trend is None:                score -= 0.05
    return round(max(0.0, min(score, 1.0)), 2)


def build_prompt(context: InputContext) -> list:
    system_message = (
        "You are a geospatial scientist specializing in remote sensing and "
        "environmental analysis with expertise in global ecosystems. "
        "STRICT RULES: "
        "Do NOT assume causation without evidence. "
        "Distinguish clearly between observation, correlation, and hypothesis. "
        "Use cautious scientific language: 'may indicate', 'suggests', 'potentially'. "
        "Consider regional climate context — arid regions naturally have low NDVI, "
        "tropical regions have high NDVI, Mediterranean regions show seasonal variation. "
        "Base conclusions ONLY on the provided metrics. "
        "Never invent data that was not provided."
    )

    # Build structured data block
    lc = context.land_cover or {}
    vb = context.vegetation_breakdown or {}

    ndvi_mean  = round(float(context.ndvi.mean()), 3)  if context.ndvi  is not None else "N/A"
    ndwi_mean  = round(float(context.ndwi.mean()), 3)  if context.ndwi  is not None else "N/A"
    ndbi_mean  = round(float(context.ndbi.mean()), 3)  if context.ndbi  is not None else "N/A"
    ndvi_trend = f"ΔNDVI={context.ndvi_trend:.3f}" if context.ndvi_trend is not None else "not available (single image)"
    water_ratio = f"{context.water_ratio*100:.1f}%" if context.water_ratio is not None else "N/A"
    aridity    = context.aridity_index if context.aridity_index else "N/A"
    confidence = compute_confidence(context)
    context.confidence_score = confidence

    anomalies_text = "\n".join(f"  - {a}" for a in (context.anomalies or [])) or "  None detected"
    climate_text   = context.retrieved_context or "No climate data provided."

    user_message = f"""
SATELLITE IMAGE ANALYSIS DATA:

Land cover (mutually exclusive):
  - Water:      {lc.get('water', 0):.2f}%
  - Vegetation: {lc.get('vegetation', 0):.2f}%
  - Urban:      {lc.get('urban', 0):.2f}%
  - Barren:     {lc.get('barren', 0):.2f}%

Vegetation breakdown:
  - Sparse (NDVI 0.1–0.25):   {vb.get('sparse_pct', 'N/A')}%
  - Moderate (NDVI 0.25–0.45): {vb.get('moderate_pct', 'N/A')}%
  - Dense (NDVI > 0.45):       {vb.get('dense_pct', 'N/A')}%

Spectral indices:
  - NDVI mean:  {ndvi_mean} (vegetation health)
  - NDWI mean:  {ndwi_mean} (water content)
  - NDBI mean:  {ndbi_mean} (built-up density)
  - NDVI trend: {ndvi_trend}
  - Water/flood coverage: {water_ratio}
  - Aridity index: {aridity}

{climate_text}

Detected anomalies:
{anomalies_text}

Confidence score: {confidence*100:.1f}%

USER QUESTION: {context.user_question}

Generate a structured scientific report with these sections:

1. LAND COVER ANALYSIS
   Describe composition and what it indicates. Reference percentages directly.

2. VEGETATION ASSESSMENT
   Use NDVI mean + breakdown (sparse/moderate/dense). Note trend if available.
   Be explicit about what NDVI values mean in context.

3. HYDROLOGICAL ASSESSMENT
   Use NDWI + water ratio. Distinguish between natural water bodies and potential flooding.

4. CLIMATE CONTEXT
   Interpret temperature, rainfall, humidity trends. Note seasonal patterns.

5. ANOMALY INTERPRETATION
   Explain each anomaly. Do NOT state causation without evidence. Use "may suggest".

6. ANSWER TO USER QUESTION
   Direct, evidence-based, cautious answer referencing specific metrics.

7. CONFIDENCE & LIMITATIONS
   State confidence score ({confidence*100:.1f}%) and specific data gaps.
   Note if temporal analysis was unavailable.
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
