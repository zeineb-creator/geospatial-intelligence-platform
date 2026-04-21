import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from geospatial_platform.input_handler import build_input_context
from geospatial_platform.image_processor import process_image
from geospatial_platform.vision_model import run_vision_module, load_vit
from geospatial_platform.data_integrator import run_data_integrator
from geospatial_platform.rag import run_rag
from geospatial_platform.llm_engine import generate_report, load_llm


def compute_temporal_ndvi(ctx_now, ctx_past):
    if ctx_now.ndvi is None or ctx_past.ndvi is None:
        return None                                      # ← explicit None

    ndvi_now  = ctx_now.ndvi
    ndvi_past = ctx_past.ndvi

    if ndvi_now.shape != ndvi_past.shape:
        from PIL import Image as PILImage
        ndvi_past_img = PILImage.fromarray(ndvi_past)
        ndvi_past = np.array(
            ndvi_past_img.resize(
                (ndvi_now.shape[1], ndvi_now.shape[0]),
                PILImage.BILINEAR
            )
        )

    delta = ndvi_now - ndvi_past
    return float(np.mean(delta)), delta


def run_pipeline(
    image_path: str,
    csv_path: str = None,
    question: str = None,
    image_path_past: str = None,   # ← new: optional second image
    extractor=None,
    vit_model=None,
    tokenizer=None,
    llm=None,
) -> dict:

    # Step 1 — Input handling
    ctx = build_input_context(
        image_path=image_path,
        csv_path=csv_path,
        question=question,
    )

    # Step 2 — Image processing
    ctx = process_image(ctx)

    # Step 3 — Temporal NDVI (if second image provided)
    if image_path_past:
        print("=== Temporal Analysis ===")
        ctx_past = build_input_context(image_path=image_path_past)
        ctx_past = process_image(ctx_past)
        ctx.ndvi_previous = ctx_past.ndvi

        if ctx.ndvi is not None and ctx_past.ndvi is not None:
            temporal_result = compute_temporal_ndvi(ctx, ctx_past)
            if temporal_result is not None:              # ← add this check
                trend_val, trend_map   = temporal_result
                ctx.ndvi_trend         = trend_val
                ctx.ndvi_trend_map     = trend_map
                print(f"  ΔNDVI (mean): {trend_val:.4f}")
                if trend_val < -0.05:
                    ctx.anomalies.append(
                        f"vegetation decline detected (ΔNDVI={trend_val:.3f})")
                elif trend_val < -0.02:
                    ctx.anomalies.append(
                        f"slight vegetation decrease (ΔNDVI={trend_val:.3f})")
                elif trend_val > 0.05:
                    ctx.anomalies.append(
                        f"vegetation improvement detected (ΔNDVI={trend_val:.3f})")
        print("=== Temporal analysis complete ===\n")

    # Step 4 — Vision module
    ctx, extractor, vit_model = run_vision_module(ctx, extractor, vit_model)

    # Step 5 — Data integration
    ctx = run_data_integrator(ctx)

    # Step 6 — RAG
    ctx = run_rag(ctx)

    # Step 7 — LLM report
    ctx = generate_report(ctx, tokenizer, llm)

    results = {
        "land_cover":           ctx.land_cover,
        "anomalies":            ctx.anomalies,
        "ndvi":                 ctx.ndvi,
        "ndwi":                 ctx.ndwi,
        "ndbi":                 ctx.ndbi,
        "ndvi_trend_map":       ctx.ndvi_trend_map,
        "ndvi_trend":           ctx.ndvi_trend,
        "image_array":          ctx.image_array,
        "final_report":         ctx.final_report,
        "retrieved_context":    ctx.retrieved_context,
        "image_format":         ctx.image_format,
        "n_bands":              ctx.n_bands,
        "image_meta":           ctx.image_meta,
        "user_question":        ctx.user_question,
        "vegetation_breakdown": ctx.vegetation_breakdown,
        "water_ratio":          ctx.water_ratio,
        "aridity_index":        ctx.aridity_index,
        "confidence_score":     ctx.confidence_score,
        "csv_df":               ctx.csv_df,
        "ecosystem":            ctx.image_meta.get("ecosystem", "Unknown"),
        "region_name":          ctx.image_meta.get("region_name", "Unknown region"),
    }

    return results, extractor, vit_model
