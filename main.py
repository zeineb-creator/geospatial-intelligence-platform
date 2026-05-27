"""
main.py — CLI / notebook pipeline runner
─────────────────────────────────────────
Fix 6: corrected all broken function calls that drifted from the actual module APIs:
  - run_data_integrator → integrate_data  (data_integrator.py)
  - load_llm removed    → Groq is initialised inside generate_report()
  - generate_report signature fixed → (ic, rag_context, anomalies)
  - run_vision_module kept; extract_vit_features used when model already loaded
  - temporal NDVI now also populates ic.ndvi_delta / ndvi_mean_t1 / ndvi_mean_t2
    so the validator and LLM prompt receive the correct values
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from geospatial_platform.input_handler   import build_input_context
from geospatial_platform.image_processor import process_image
from geospatial_platform.vision_model    import run_vision_module, load_vit
from geospatial_platform.data_integrator import (
    integrate_data, build_climate_summary, populate_convenience_fields
)
from geospatial_platform.rag             import run_rag
from geospatial_platform.llm_engine      import generate_report   # no load_llm needed
from geospatial_platform.validator       import validate_inputs, build_reliability_report


# ── Temporal NDVI helper ──────────────────────────────────────────────────────

def compute_temporal_ndvi(ctx_now, ctx_past):
    """
    Compute pixel-wise ΔNDVI between two processed InputContext objects.
    Populates ctx_now with ndvi_delta, ndvi_mean_t1, ndvi_mean_t2, ndvi_trend_map.
    Returns (mean_delta, delta_map) or None on failure.
    """
    try:
        if ctx_now.ndvi is None or ctx_past.ndvi is None:
            print("  [INFO] NDVI not available on one or both contexts — skipping temporal.")
            return None

        ndvi_now  = ctx_now.ndvi.astype(np.float32)
        ndvi_past = ctx_past.ndvi.astype(np.float32)

        if ndvi_now.shape != ndvi_past.shape:
            from scipy.ndimage import zoom
            zoom_h    = ndvi_now.shape[0] / ndvi_past.shape[0]
            zoom_w    = ndvi_now.shape[1] / ndvi_past.shape[1]
            ndvi_past = zoom(ndvi_past, (zoom_h, zoom_w), order=1)
            print(f"  Resized past NDVI → {ndvi_past.shape}")

        delta = ndvi_now - ndvi_past
        valid = delta[~np.isnan(delta)]
        if len(valid) == 0:
            print("  [WARNING] No valid pixels for ΔNDVI — skipping.")
            return None

        mean_delta = float(np.nanmean(delta))
        mean_t1    = float(np.nanmean(ndvi_past))
        mean_t2    = float(np.nanmean(ndvi_now))
        print(f"  ΔNDVI: mean={mean_delta:.4f}  t1_mean={mean_t1:.4f}  t2_mean={mean_t2:.4f}")

        # Populate all temporal fields on the current context
        ctx_now.ndvi_previous = ndvi_past
        ctx_now.ndvi_trend     = mean_delta
        ctx_now.ndvi_trend_map = delta
        ctx_now.ndvi_delta     = round(mean_delta, 4)
        ctx_now.ndvi_mean_t1   = round(mean_t1, 4)
        ctx_now.ndvi_mean_t2   = round(mean_t2, 4)

        return mean_delta, delta

    except Exception as e:
        print(f"  [WARNING] Temporal NDVI failed: {e}")
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    csv_path: str = None,
    question: str = None,
    image_path_past: str = None,
    extractor=None,
    vit_model=None,
) -> tuple:
    """
    Full pipeline runner for CLI / Kaggle notebook use.
    Returns (results_dict, extractor, vit_model) so the caller can
    cache the ViT model across multiple runs.
    """

    # ── Step 1: Input handling ────────────────────────────────────────────────
    ctx = build_input_context(
        image_path=image_path,
        csv_path=csv_path,
        question=question,
    )

    # ── Step 2: Image processing ──────────────────────────────────────────────
    ctx = process_image(ctx)

    # ── Step 3: Temporal NDVI ─────────────────────────────────────────────────
    if image_path_past:
        print("=== Temporal Analysis ===")
        try:
            ctx_past = build_input_context(image_path=image_path_past)
            ctx_past = process_image(ctx_past)

            temporal_result = compute_temporal_ndvi(ctx, ctx_past)
            if temporal_result is not None:
                mean_delta, _ = temporal_result
                # Anomaly flags
                if mean_delta < -0.05:
                    ctx.anomalies.append(f"vegetation decline detected (ΔNDVI={mean_delta:.3f})")
                elif mean_delta < -0.02:
                    ctx.anomalies.append(f"slight vegetation decrease (ΔNDVI={mean_delta:.3f})")
                elif mean_delta > 0.05:
                    ctx.anomalies.append(f"vegetation improvement detected (ΔNDVI={mean_delta:+.3f})")
                elif mean_delta > 0.02:
                    ctx.anomalies.append(f"slight vegetation increase (ΔNDVI={mean_delta:+.3f})")
            else:
                print("  [INFO] Temporal NDVI not computable — continuing without.")
        except Exception as e:
            print(f"  [WARNING] Temporal analysis failed: {e}")
        print("=== Temporal analysis complete ===\n")

    # ── Step 4: Vision module (ViT embedding + anomalies) ────────────────────
    try:
        ctx, extractor, vit_model = run_vision_module(ctx, extractor, vit_model)
    except Exception as e:
        print(f"  [WARNING] Vision module failed: {e}")
        if extractor is None or vit_model is None:
            extractor, vit_model = load_vit()

    # ── Step 5: Validator ─────────────────────────────────────────────────────
    try:
        flags = validate_inputs(ctx)
        reliability = build_reliability_report(flags)
        ctx.image_meta["reliability_report"] = reliability
        print(reliability)
    except Exception as e:
        print(f"  [WARNING] Validator failed: {e}")

    # ── Step 6: Data integration ──────────────────────────────────────────────
    try:
        ctx = integrate_data(ctx)
        if ctx.csv_df is not None:
            ctx.climate_summary = build_climate_summary(ctx.csv_df)
            populate_convenience_fields(ctx)
    except Exception as e:
        print(f"  [WARNING] Data integrator failed: {e}")

    # ── Step 7: RAG ───────────────────────────────────────────────────────────
    try:
        ctx = run_rag(ctx)
        # Normalise field name so downstream code always reads rag_context
        if not getattr(ctx, 'rag_context', None):
            ctx.rag_context = getattr(ctx, 'retrieved_context', '') or ''
    except Exception as e:
        print(f"  [WARNING] RAG failed: {e}")
        ctx.rag_context = ""

    # ── Step 8: Confidence score ──────────────────────────────────────────────
    if ctx.confidence_score is None:
        score = 0.0
        if ctx.ndvi is not None:                      score += 20
        if ctx.ndwi is not None:                      score += 15
        if ctx.ndbi is not None:                      score += 10
        if ctx.aridity_index is not None:             score += 10
        if getattr(ctx, 'csv_df', None) is not None: score += 20
        if ctx.ndvi_delta is not None:                score += 15
        ctx.confidence_score = min(85.0, score)

    # ── Step 9: LLM report ────────────────────────────────────────────────────
    # Fix 6: generate_report(ic, rag_context, anomalies) — no tokenizer/llm args
    try:
        ctx.report = generate_report(
            ctx,
            ctx.rag_context or "",
            ctx.anomalies  or [],
        )
        ctx.final_report = ctx.report  # keep backward-compat alias
    except Exception as e:
        print(f"  [WARNING] LLM report failed: {e}")
        ctx.report = ctx.final_report = f"Report generation failed: {e}"

    # ── Package results ───────────────────────────────────────────────────────
    results = {
        "land_cover":           ctx.land_cover         or {},
        "anomalies":            ctx.anomalies          or [],
        "ndvi":                 ctx.ndvi,
        "ndwi":                 ctx.ndwi,
        "ndbi":                 ctx.ndbi,
        "ndvi_mean":            ctx.ndvi_mean,
        "ndwi_mean":            ctx.ndwi_mean,
        "ndbi_mean":            ctx.ndbi_mean,
        "ndvi_delta":           ctx.ndvi_delta,
        "ndvi_mean_t1":         ctx.ndvi_mean_t1,
        "ndvi_mean_t2":         ctx.ndvi_mean_t2,
        "ndvi_trend_map":       getattr(ctx, "ndvi_trend_map", None),
        "image_array":          ctx.image_array,
        "report":               ctx.report,
        "final_report":         ctx.final_report,
        "retrieved_context":    ctx.rag_context,
        "image_format":         ctx.image_format,
        "n_bands":              ctx.n_bands,
        "image_meta":           ctx.image_meta,
        "user_question":        ctx.user_question,
        "aridity_index":        ctx.aridity_index,
        "confidence_score":     ctx.confidence_score,
        "climate_summary":      ctx.climate_summary,
        "csv_df":               ctx.csv_df,
        "ecosystem":            ctx.ecosystem or ctx.image_meta.get("ecosystem",   "Unknown"),
        "region_name":          ctx.region    or ctx.image_meta.get("region_name", "Unknown region"),
        "reliability_report":   ctx.image_meta.get("reliability_report", ""),
    }

    return results, extractor, vit_model


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Geospatial Intelligence Platform — CLI")
    parser.add_argument("image",          help="Path to primary satellite image (GeoTIFF)")
    parser.add_argument("--past",         help="Path to past image for temporal comparison")
    parser.add_argument("--csv",          help="Path to NASA POWER climate CSV")
    parser.add_argument("--question",     default=None, help="Custom analysis question")
    args = parser.parse_args()

    results, _, _ = run_pipeline(
        image_path      = args.image,
        csv_path        = args.csv,
        question        = args.question,
        image_path_past = args.past,
    )

    print("\n── RESULTS ──────────────────────────────────────")
    print(f"Region     : {results['region_name']}")
    print(f"Ecosystem  : {results['ecosystem']}")
    print(f"Land cover : {results['land_cover']}")
    print(f"Confidence : {results['confidence_score']}%")
    print(f"ΔNDVI      : {results['ndvi_delta']}")
    print("\n── REPORT ───────────────────────────────────────")
    print(results["report"])
