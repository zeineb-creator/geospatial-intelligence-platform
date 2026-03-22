
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional

@dataclass
class InputContext:
    # Image data
    image_array: np.ndarray          # shape: (bands, H, W)
    image_meta: dict                 # resolution, CRS, format, etc.
    image_format: str                # 'geotiff' or 'rgb'
    n_bands: int                     # 1, 3, or 13 depending on source

    # Optional CSV
    csv_df: Optional[pd.DataFrame] = None
    csv_summary: Optional[dict] = None

    # Optional user question
    user_question: str = "Provide a full scientific interpretation of this image and data."

    # Computed later by downstream modules (start as None)
    ndvi: Optional[np.ndarray] = None
    ndwi: Optional[np.ndarray] = None
    ndbi: Optional[np.ndarray] = None
    land_cover: Optional[dict] = None      # e.g. {"vegetation": 0.4, "urban": 0.22}
    anomalies: Optional[list] = field(default_factory=list)
    retrieved_context: Optional[str] = None
    final_report: Optional[str] = None
