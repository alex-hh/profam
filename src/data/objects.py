from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# want to be consistent with fields in parquet files so we can load from there
# TODO: look into how openai evals uses data classes or similar
# TODO: consider how to represent masks
@dataclass
class ProteinDocument:
    identifier: str
    sequences: List[str]
    accessions: List[str]
    backbone_coords: Optional[np.ndarray]
    prompt_indices: Optional[List[int]]
