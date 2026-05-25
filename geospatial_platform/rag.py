
import numpy as np
import pandas as pd
import sys
sys.path.append("/kaggle/working")

from geospatial_platform.context import InputContext


# Retrieval rules: vision signal → relevant CSV categories to fetch
RETRIEVAL_RULES = {
    "vegetation":    ["rainfall", "ndvi", "drought", "humidity", "temperature"],
    "water":         ["flood", "rainfall", "humidity"],
    "urban":         ["temperature", "elevation"],
    "barren":        ["drought", "rainfall", "temperature"],
    "drought":       ["rainfall", "temperature", "ndvi"],
    "flood":         ["rainfall", "flood", "humidity"],
    "temperature":   ["temperature", "humidity"],
    "default":       ["rainfall", "temperature", "ndvi"],
}


def detect_active_signals(context: InputContext) -> list:
    """
    Identify which environmental signals are active based on
    vision results, index values, and detected anomalies.
    Returns a list of signal keywords to drive retrieval.
    """
    signals = []

    # From land cover
    if context.land_cover:
        for cls, pct in context.land_cover.items():
            if pct > 10.0:
                signals.append(cls)

    # From NDVI value
    if context.ndvi is not None:
        mean_ndvi = float(context.ndvi.mean())
        if mean_ndvi < 0.2:
            signals.append("drought")
        if mean_ndvi > 0.5:
            signals.append("vegetation")

    # From NDWI value
    if context.ndwi is not None:
        mean_ndwi = float(context.ndwi.mean())
        if mean_ndwi > 0.2:
            signals.append("flood")

    # From anomaly strings
    for anomaly in (context.anomalies or []):
        anomaly_lower = anomaly.lower()
        for keyword in RETRIEVAL_RULES:
            if keyword in anomaly_lower:
                signals.append(keyword)

    # From user question
    question_lower = context.user_question.lower()
    for keyword in RETRIEVAL_RULES:
        if keyword in question_lower:
            signals.append(keyword)

    # Deduplicate while preserving order
    seen = set()
    unique_signals = []
    for s in signals:
        if s not in seen:
            seen.add(s)
            unique_signals.append(s)

    return unique_signals if unique_signals else ["default"]


def retrieve_relevant_columns(
    signals: list,
    env_summary: dict,
) -> dict:
    """
    Given active signals and the environmental summary,
    retrieve only the columns that are relevant.
    Returns a filtered subset of env_summary.
    """
    # Build the set of relevant categories from all active signals
    relevant_categories = set()
    for signal in signals:
        categories = RETRIEVAL_RULES.get(signal, RETRIEVAL_RULES["default"])
        relevant_categories.update(categories)

    # Filter env_summary to only matching columns
    retrieved = {}
    for col, stats in env_summary.items():
        if stats["category"] in relevant_categories:
            retrieved[col] = stats

    # If nothing matched, return everything (safe fallback)
    if not retrieved:
        retrieved = env_summary

    return retrieved


def score_and_rank(retrieved: dict) -> dict:
    """
    Rank retrieved columns by relevance score.
    Score = trend strength + anomaly magnitude.
    Ensures the most informative variables appear first in the prompt.
    """
    scored = {}
    for col, stats in retrieved.items():
        score = 0.0

        # Trend contributes to score
        if stats["trend"] == "decreasing":
            score += 1.5
        elif stats["trend"] == "increasing":
            score += 1.0

        # Distance from mean contributes
        mean = stats["mean"]
        latest = stats["latest"]
        if mean != 0:
            deviation = abs(latest - mean) / abs(mean)
            score += deviation

        scored[col] = (score, stats)

    # Sort by score descending
    ranked = dict(
        sorted(scored.items(), key=lambda x: x[1][0], reverse=True)
    )

    # Return just the stats dicts in ranked order
    return {col: stats for col, (_, stats) in ranked.items()}


def build_rag_context(
    signals: list,
    retrieved: dict,
    anomalies: list,
    user_question: str,
) -> str:
    """
    Format the retrieved data into a structured text block
    ready for injection into the LLM prompt.
    """
    lines = []

    lines.append(f"User question: {user_question}")
    lines.append("")
    lines.append(f"Active environmental signals: {', '.join(signals)}")
    lines.append("")

    if retrieved:
        lines.append("Most relevant data retrieved:")
        for col, stats in retrieved.items():
            lines.append(
                f"  - {col} ({stats['category']}): "
                f"latest={stats['latest']}, "
                f"mean={stats['mean']}, "
                f"trend={stats['trend']}, "
                f"range=[{stats['min']}, {stats['max']}]"
            )
    else:
        lines.append("No CSV data available for retrieval.")

    lines.append("")

    if anomalies:
        lines.append("Detected anomalies (vision + data):")
        for a in anomalies:
            lines.append(f"  - {a}")
    else:
        lines.append("No anomalies detected.")

    return "\n".join(lines)

    
def run_rag(context: InputContext) -> InputContext:
    """
    Main entry point for the RAG module.
    Detects signals, retrieves relevant data, builds context string.
    """
    print("=== RAG Module ===")

    # Step 1: detect what signals are active
    signals = detect_active_signals(context)
    print(f"  Active signals   : {signals}")

    # Step 2: retrieve relevant CSV columns
    env_summary = {}
    if context.csv_summary and "env_summary" in context.csv_summary:
        env_summary = context.csv_summary["env_summary"]

    retrieved = retrieve_relevant_columns(signals, env_summary)
    print(f"  Retrieved columns: {list(retrieved.keys())}")

    # Step 3: rank by relevance
    ranked = score_and_rank(retrieved)
    print(f"  Ranked columns   : {list(ranked.keys())}")

    # Step 4: build context string for LLM
    rag_text = build_rag_context(
        signals=signals,
        retrieved=ranked,
        anomalies=context.anomalies or [],
        user_question=context.user_question,
    )

    context.retrieved_context = rag_text

    print("=== RAG complete ===\n")
    print("--- Retrieved context block ---")
    print(rag_text)
    print("-------------------------------\n")

    return context

def retrieve_context(context: InputContext) -> InputContext:
    """
    Wrapper function that matches what app.py expects.
    Calls the existing run_rag function.
    """
    return run_rag(context)
    
