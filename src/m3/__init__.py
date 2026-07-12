"""m3 -- multimodal, multi-batch, condition-aware single-cell model."""

from m3.io import read_matrix, from_anndata, read_h5ad, read_h5
from m3._concat import concat
from m3 import datasets

__version__ = "0.3.0.dev0"

__all__ = [
    "read_matrix", "from_anndata", "read_h5ad", "read_h5", "concat",
    "M3", "Attribution", "M3CapabilityError",
    "datasets",
]

# The model and its readouts need torch; import them lazily so that `import m3`
# and the torch-free readers (io) and bundled datasets stay usable without a
# torch install. Accessing m3.M3 (etc.) triggers the torch-backed import.
_LAZY = {"M3", "Attribution", "M3CapabilityError"}


def __getattr__(name):
    if name in _LAZY:
        from m3 import _model
        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
