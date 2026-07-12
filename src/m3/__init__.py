"""m3 -- multimodal, multi-batch, condition-aware single-cell model."""

from m3.io import read_matrix, from_anndata, read_h5ad, read_h5
from m3._concat import concat
from m3._model import M3, Attribution, M3CapabilityError
from m3 import datasets

__version__ = "0.3.0.dev0"

__all__ = [
    "read_matrix", "from_anndata", "read_h5ad", "read_h5", "concat",
    "M3", "Attribution", "M3CapabilityError",
    "datasets",
]
