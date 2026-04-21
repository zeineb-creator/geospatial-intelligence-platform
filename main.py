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
    """
    Compute NDVI trend between two images.
    Returns (mean_delta, delta_map) or None if not computable.
    """
    try:
        if ctx_now.ndvi is None or ctx_past.ndvi is None:
            return None

        ndvi_now  = ctx_now.ndvi
        ndvi_past = ctx_past.ndvi

        # Resize if shapes differ
        if ndvi_now.shape != ndvi_past.shape:
            from PIL import Image as PILImage
            pil_past  = PILImage.fromarray(ndvi_past)
            ndvi_past = np.array(
                pil_past.resize(
                    (ndvi_now.shape[1], ndvi_now.shape[0]),
                    PILImage.BILINEAR
                )
            )

        delta = ndvi_now - ndvi_past
        return float(np.mean(delta)), delta

    except Exception as e:
        print(f"  [WARNING] Temporal NDVI failed: {e}")
        return None


def run_pipeline(
    image_path: str,
    csv_path: str = None,
    question: str = None,
    image_path_past: str = None,
    extractor=None,
    vit_model=None,
    tokenizer=None,
    llm=None,
) -> tuple:

    # Step 1 — Input handling
    ctx = build_input_context(
        image_path=image_path,
        csv_path=csv_path,
        question=question,
    )

    # Step 2 — Image processing
    ctx = process_image(ctx)

    # Step 3 — Temporal NDVI (only if second image provided)
    if image_path_past:
        print("=== Temporal Analysis ===")
        try:
            ctx_past = build_input_context(image_path=image_path_past)
            ctx_past = process_image(ctx_past)
            ctx.ndvi_previous = ctx_past.ndvi

            temporal_result = compute_temporal_ndvi(ctx, ctx_past)

            if temporal_result is not None:
                trend_val, trend_map = temporal_result
                ctx.ndvi_trend     = trend_val
                ctx.ndvi_trend_map = trend_map
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
                elif trend_val > 0.02:
                    ctx.anomalies.append(
                        f"slight vegetation increase (ΔNDVI={trend_val:.3f})")
            else:
                print("  [INFO] Temporal NDVI not computable — skipping.")

        except Exception as e:
            print(f"  [WARNING] Temporal analysis failed: {e}. Continuing without it.")

        print("=== Temporal analysis complete ===\n")

    # Step 4 — Vision module
    try:
        ctx, extractor, vit_model = run_vision_module(ctx, extractor, vit_model)
    except Exception as e:
        print(f"  [WARNING] Vision module failed: {e}")

    # Step 5 — Data integration
    try:
        ctx = run_data_integrator(ctx)
    except Exception as e:
        print(f"  [WARNING] Data integrator failed: {e}")

    # Step 6 — RAG
    try:
        ctx = run_rag(ctx)
    except Exception as e:
        print(f"  [WARNING] RAG failed: {e}")

    # Step 7 — LLM report
    try:
        ctx = generate_report(ctx, tokenizer, llm)
    except Exception as e:
        print(f"  [WARNING] LLM failed: {e}")
        ctx.final_report = f"Report generation failed: {e}"

    # Package results
    results = {
        "land_cover":           ctx.land_cover or {},
        "anomalies":            ctx.anomalies  or [],
        "ndvi":                 ctx.ndvi,
        "ndwi":                 ctx.ndwi,
        "ndbi":                 ctx.ndbi,
        "ndvi_trend_map":       ctx.ndvi_trend_map if hasattr(ctx, "ndvi_trend_map") else None,
        "ndvi_trend":           ctx.ndvi_trend     if hasattr(ctx, "ndvi_trend")     else None,
        "image_array":          ctx.image_array,
        "final_report":         ctx.final_report,
        "retrieved_context":    ctx.retrieved_context,
        "image_format":         ctx.image_format,
        "n_bands":              ctx.n_bands,
        "image_meta":           ctx.image_meta,
        "user_question":        ctx.user_question,
        "vegetation_breakdown": ctx.vegetation_breakdown if hasattr(ctx, "vegetation_breakdown") else None,
        "water_ratio":          ctx.water_ratio          if hasattr(ctx, "water_ratio")          else None,
        "aridity_index":        ctx.aridity_index        if hasattr(ctx, "aridity_index")        else None,
        "confidence_score":     ctx.confidence_score     if hasattr(ctx, "confidence_score")     else None,
        "csv_df":               ctx.csv_df,
        "ecosystem":            ctx.image_meta.get("ecosystem",    "Unknown"),
        "region_name":          ctx.image_meta.get("region_name",  "Unknown region"),
    }

    return results, extractor, vit_model
