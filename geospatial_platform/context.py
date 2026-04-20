from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional

@dataclass
class InputContext:
    # Image data
    image_array: np.ndarray
    image_meta: dict
    image_format: str
    n_bands: int

    # Optional CSV
    csv_df: Optional[pd.DataFrame] = None
    csv_summary: Optional[dict] = None

    # Optional user question
    user_question: str = "Provide a full scientific interpretation of this image and data."

    # Spectral indices
    ndvi: Optional[np.ndarray] = None
    ndwi: Optional[np.ndarray] = None
    ndbi: Optional[np.ndarray] = None

    # Temporal (only when second image provided)
    ndvi_previous: Optional[np.ndarray] = None
    ndvi_trend: Optional[float] = None

    # Derived metrics
    water_ratio: Optional[float] = None
    aridity_index: Optional[float] = None
    vegetation_breakdown: Optional[dict] = None

    # Results
    land_cover: Optional[dict] = None
    anomalies: Optional[list] = field(default_factory=list)
    retrieved_context: Optional[str] = None
    final_report: Optional[str] = None
    confidence_score: Optional[float] = None
