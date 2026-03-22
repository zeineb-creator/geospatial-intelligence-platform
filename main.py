
import sys
sys.path.append("/kaggle/working")

from geospatial_platform.input_handler import build_input_context
from geospatial_platform.image_processor import process_image
from geospatial_platform.vision_model import run_vision_module, load_vit
from geospatial_platform.data_integrator import run_data_integrator
from geospatial_platform.rag import run_rag
from geospatial_platform.llm_engine import generate_report, load_llm


def run_pipeline(
    image_path: str,
    csv_path: str = None,
    question: str = None,
    hf_token: str = None,
    extractor=None,
    vit_model=None,
    tokenizer=None,
    llm=None,
) -> dict:
    """
    Full pipeline orchestrator.
    Accepts file paths and optional pre-loaded models.
    Returns a results dict with everything needed for the UI.
    """

    # Step 1 — Input handling
    ctx = build_input_context(
        image_path=image_path,
        csv_path=csv_path,
        question=question,
    )

    # Step 2 — Image processing
    ctx = process_image(ctx)

    # Step 3 — Vision module
    ctx, extractor, vit_model = run_vision_module(ctx, extractor, vit_model)

    # Step 4 — Data integration
    ctx = run_data_integrator(ctx)

    # Step 5 — RAG
    ctx = run_rag(ctx)

    # Step 6 — LLM report generation
    ctx = generate_report(ctx, tokenizer, llm)

    # Package results for the UI
    results = {
        "land_cover":        ctx.land_cover,
        "anomalies":         ctx.anomalies,
        "ndvi":              ctx.ndvi,
        "ndwi":              ctx.ndwi,
        "image_array":       ctx.image_array,
        "final_report":      ctx.final_report,
        "retrieved_context": ctx.retrieved_context,
        "image_format":      ctx.image_format,
        "n_bands":           ctx.n_bands,
        "image_meta":        ctx.image_meta,
        "user_question":     ctx.user_question,
    }

    return results, extractor, vit_model
