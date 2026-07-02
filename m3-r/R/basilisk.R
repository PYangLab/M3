# basilisk environment bundling the m3 Python engine's dependencies.
#
# Version notes:
#  * m3-sc requires Python >= 3.10. We pin a validated stack
#    (torch 2.1.2 from PyPI, which ships CUDA and falls back to CPU when no GPU
#    is present) so a single pin gives GPU on capable hosts and CPU elsewhere.
#  * The engine itself needs numpy/scipy/pandas/h5py + torch + captum (integrated
#    gradients) + scanpy/anndata/scikit-learn (data handling). umap-learn is
#    pulled in for the tutorials' projections so Python and R share the SAME UMAP
#    implementation (no uwot-vs-umap-learn rotation).
#  * Identical numerics to the Python package are validated as bit-identical
#    parity in the SAME env + device + seed.

#' @importFrom basilisk BasiliskEnvironment
m3_env <- basilisk::BasiliskEnvironment(
  envname  = "env-m3",
  pkgname  = "m3",
  packages = c(            # conda-forge
    "python=3.10",
    "numpy=1.23.5",
    "pandas=1.5.3",
    "scipy=1.10.1",
    "h5py=3.8.0"
  ),
  pip = c(                 # PyPI (torch ships CUDA; CPU fallback when no GPU)
    "torch==2.1.2",
    "captum==0.7.0",
    "anndata==0.10.8",
    "scanpy==1.9.6",
    "scikit-learn==1.3.2",
    "umap-learn==0.5.5",
    "tqdm==4.66.1"
  )
)
