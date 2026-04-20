import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from geospatial_platform.context import InputContext


MODEL_NAME = "llama-3.1-8b-instant"


def load_llm(hf_token: str = None):
    """
    Initialize Groq client.
    Returns a (None, client) tuple to keep the same interface as before.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to Streamlit secrets.")

    client = Groq(api_key=api_key)
    print(f"  Groq client initialized. Model: {MODEL_NAME}")
    return None, client


def build_prompt(context: InputContext) -> list:
    """
    Construct the full structured prompt from the retrieved context.
    """
    system_message = (
        "You are a scientific geospatial analyst specializing in Mediterranean ecosystems. "
        "You interpret satellite imagery and environmental data to produce "
        "accurate, evidence-based reports. "
        "Be precise, use scientific terminology, and ground every claim in the data provided. "
        "Consider seasonal context when interpreting NDVI values — Mediterranean vegetation "
        "naturally shows lower NDVI in summer due to drought adaptation. "
        "Structure your response with clear sections."
    )

    vision_block = "Visual analysis (satellite image):\n"
    if context.land_cover:
        for cls, pct in context.land_cover.items():
            if pct > 0:
                vision_block += f"  - {cls}: {pct}% coverage\n"
    else:
        vision_block += "  - No land cover data available\n"

    if context.anomalies:
        vision_block += "\nDetected anomalies:\n"
        for a in context.anomalies:
            vision_block += f"  - {a}\n"

    data_block = context.retrieved_context or "No environmental data available."

    user_message = f"""
{vision_block}

{data_block}

Based on the satellite image analysis and environmental data above, please provide:

1. LAND COVER SUMMARY
   Describe the land cover composition and what it indicates about this region.

2. ENVIRONMENTAL ASSESSMENT
   Analyze the environmental conditions using the data provided.
   Identify any stress factors, trends, or risks.

3. ANOMALY EXPLANATION
   Explain each detected anomaly with reference to the supporting data.

4. ANSWER TO USER QUESTION
   Question: {context.user_question}
   Provide a direct, evidence-based answer.

5. CONFIDENCE & LIMITATIONS
   State your confidence level and any limitations due to data availability.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message.strip()},
    ]


def generate_report(
    context: InputContext,
    tokenizer,       # unused — kept for interface compatibility
    llm,             # this is now the Groq client
    max_new_tokens: int = 600,
) -> InputContext:
    """
    Generate report using Groq API.
    """
    print("=== LLM Reasoning Engine (Groq) ===")

    messages = build_prompt(context)

    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )

    report = response.choices[0].message.content
    context.final_report = report

    print("  Report generated via Groq.")
    print("=== LLM complete ===\n")

    return context
