
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
    rgb = np.clip(rgb, 0, 1)
    if ndvi is not None:
        ndvi_norm = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-8)
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
        scores["vegetation"] = round(float((context.ndvi > 0.2).sum()) / total_pixels * 100, 2)
    if context.ndwi is not None:
        scores["water"] = round(float((context.ndwi > 0.0).sum()) / total_pixels * 100, 2)
    if context.ndbi is not None:
        scores["urban"] = round(float((context.ndbi > 0.0).sum()) / total_pixels * 100, 2)

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
        if mean_ndvi < 0.1:
            anomalies.append("severe vegetation stress detected (NDVI < 0.1)")
        elif mean_ndvi < 0.2:
            anomalies.append("moderate vegetation decline detected (NDVI < 0.2)")
        if mean_ndvi > 0.6:
            anomalies.append("high vegetation density detected (NDVI > 0.6)")
    if context.ndwi is not None:
        if float(context.ndwi.mean()) > 0.3:
            anomalies.append("significant water presence detected (NDWI > 0.3)")

    return scores, anomalies


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

    land_cover, anomalies = classify_land_cover(context, cls_embedding)
    context.land_cover = land_cover
    context.anomalies  = anomalies

    print(f"  Land cover : {land_cover}")
    print(f"  Anomalies  : {anomalies if anomalies else 'none detected'}")

    context.image_meta["cls_embedding"] = cls_embedding
    print("=== Vision module complete ===\n")
    return context, extractor, model
