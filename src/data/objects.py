import numpy as np
from typing import List, Optional
from dataclasses import dataclass


# want to be consistent with fields in parquet files so we can load from there
# TODO: look into how openai evals uses data classes or similar
# TODO: consider how to represent masks
@dataclass
class ProteinDocument:
    identifier: str
    sequences: List[str]
    accessions: List[str]
    backbone_coords: Optional[np.ndarray]
