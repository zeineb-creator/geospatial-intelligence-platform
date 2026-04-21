
import numpy as np
import torch
import sys
sys.path.append("/kaggle/working")

from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from geospatial_platform.context import InputContext


LAND_COVER_CLASSES = ["vegetation", "water", "urban", "barren", "unknown"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vit(model_name: str = "google/vit-base-patch16-224"):
    print(f"  Loading ViT: {model_name}")
    extractor = ViTImageProcessor.from_pretrained(model_name)
    model     = ViTModel.from_pretrained(model_name)
    model.eval()
    model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    print(f"  Device: {DEVICE}")
    print(f"  Parameters frozen: yes")
    return extractor, model


def prepare_image_for_vit(image_array: np.ndarray, ndvi: np.ndarray = None) -> np.ndarray:
    rgb = image_array[:3]
    rgb = np.transpose(rgb, (1, 2, 0))
    # Replace NaN with 0 before clipping
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb, 0, 1)

    if ndvi is not None:
        ndvi_clean = np.nan_to_num(ndvi, nan=0.0)
        ndvi_norm  = (ndvi_clean - ndvi_clean.min()) / (ndvi_clean.max() - ndvi_clean.min() + 1e-8)
        rgb[:, :, 1] = 0.7 * rgb[:, :, 1] + 0.3 * ndvi_norm

    return (rgb * 255).astype(np.uint8)


def extract_features(image_uint8: np.ndarray, extractor, model) -> np.ndarray:
    pil_image = Image.fromarray(image_uint8)
    inputs = extractor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding


def classify_land_cover(context: InputContext, cls_embedding: np.ndarray) -> tuple:
    scores = {cls: 0.0 for cls in LAND_COVER_CLASSES}
    total_pixels = context.image_array.shape[1] * context.image_array.shape[2]

    if context.ndvi is not None:
        scores["vegetation"] = round(
            float((context.ndvi > 0.1).sum()) / total_pixels * 100, 2)
    if context.ndwi is not None:
        scores["water"] = round(
            float((context.ndwi > 0.0).sum()) / total_pixels * 100, 2)
    if context.ndbi is not None:
        scores["urban"] = round(
            float((context.ndbi > 0.05).sum()) / total_pixels * 100, 2)

    if context.ndvi is None:
        embedding_norm = float(np.linalg.norm(cls_embedding))
        if embedding_norm > 200:
            scores["vegetation"] = 35.0
            scores["urban"]      = 25.0
            scores["water"]      = 10.0
        else:
            scores["barren"]  = 50.0
            scores["unknown"] = 30.0

    anomalies = []
    if context.ndvi is not None:
        mean_ndvi = float(context.ndvi.mean())
    
        # Only describe current state — never imply change without temporal data
        if mean_ndvi < 0.05:
            anomalies.append(
                f"very low vegetation density observed (mean NDVI={mean_ndvi:.3f}) "
                f"— temporal change unknown"
            )
        elif mean_ndvi < 0.15:
            anomalies.append(
                f"low vegetation density observed (mean NDVI={mean_ndvi:.3f}) "
                f"— temporal change unknown"
            )
        elif mean_ndvi > 0.5:
            anomalies.append(
                f"high vegetation density observed (mean NDVI={mean_ndvi:.3f})"
            )
    
    if context.ndwi is not None:
        mean_ndwi = float(context.ndwi.mean())
        if mean_ndwi > 0.3:
            anomalies.append(
                f"significant water presence (mean NDWI={mean_ndwi:.3f})"
            )
        elif mean_ndwi < -0.3:
            anomalies.append(
                f"very low water content (mean NDWI={mean_ndwi:.3f}) "
                f"— possible dry conditions"
            )
    
    if context.ndbi is not None:
        mean_ndbi = float(context.ndbi.mean())
        if mean_ndbi > 0.2:
            anomalies.append(
                f"high built-up density observed (mean NDBI={mean_ndbi:.3f})"
            )
def run_vision_module(context: InputContext, extractor=None, model=None) -> tuple:
    print("=== Vision Module ===")
    if extractor is None or model is None:
        extractor, model = load_vit()

    image_uint8 = prepare_image_for_vit(context.image_array, context.ndvi)
    print(f"  Input to ViT : {image_uint8.shape} dtype={image_uint8.dtype}")

    print("  Extracting CLS embedding...")
    cls_embedding = extract_features(image_uint8, extractor, model)
    print(f"  Embedding shape : {cls_embedding.shape}")
    print(f"  Embedding norm  : {np.linalg.norm(cls_embedding):.2f}")

    # Use image processor land cover if already computed (more accurate)
    if context.land_cover and len(context.land_cover) > 0:
        land_cover = context.land_cover
        print(f"  Using image processor land cover: {land_cover}")
    else:
        land_cover, _ = classify_land_cover(context, cls_embedding)

    # Always compute anomalies from indices
    _, anomalies = classify_land_cover(context, cls_embedding)

    context.land_cover = land_cover
    context.anomalies  = anomalies

    print(f"  Land cover : {land_cover}")
    print(f"  Anomalies  : {anomalies if anomalies else 'none detected'}")

    context.image_meta["cls_embedding"] = cls_embedding
    print("=== Vision module complete ===\n")
    return context, extractor, model
