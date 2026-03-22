
import os
import sys
import torch
sys.path.append("/kaggle/working")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from geospatial_platform.context import InputContext


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def load_llm(hf_token: str = None):
    """
    Load LLaMA 3.2 3B Instruct in 4-bit quantization.
    4-bit keeps it well within T4 VRAM limits (~6GB vs 16GB available).
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found. Set it via Kaggle secrets.")

    print(f"  Loading LLM: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=token,
    )

    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=token,
        quantization_config=bnb_config,
        device_map="auto",
    )

    llm.eval()
    print("  LLM loaded and quantized (4-bit).")
    return tokenizer, llm


def build_prompt(context: InputContext) -> list:
    """
    Construct the full structured prompt from the retrieved context.
    Uses LLaMA 3 chat format with a system + user message.
    """
    system_message = (
        "You are a scientific geospatial analyst. "
        "You interpret satellite imagery and environmental data to produce "
        "accurate, evidence-based reports. "
        "Be precise, use scientific terminology, and ground every claim in the data provided. "
        "Structure your response with clear sections."
    )

    # Build vision summary block
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

    # Build data block from RAG
    data_block = context.retrieved_context or "No environmental data available."

    # Assemble user message
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
    tokenizer,
    llm,
    max_new_tokens: int = 600,
) -> InputContext:
    """
    Run the LLM on the structured prompt and store the report in context.
    """
    print("=== LLM Reasoning Engine ===")

    messages = build_prompt(context)

    # Apply chat template — handle both tensor and BatchEncoding return types
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids.to(DEVICE)
    else:
        input_ids = encoded.to(DEVICE)

    print(f"  Prompt tokens : {input_ids.shape[1]}")
    print("  Generating report...")

    with torch.no_grad():
        output_ids = llm.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[1]:]
    report = tokenizer.decode(new_tokens, skip_special_tokens=True)

    context.final_report = report
    print("  Report generated.")
    print("=== LLM complete ===\n")

    return context
