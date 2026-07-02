# m3 (R)

R interface to **m3** — a multimodal (RNA / ADT / ATAC), multi-batch,
condition-aware single-cell model: batch-corrected, condition-disentangled cell
embeddings; an adversarial **donor-level disease predictor**; end-to-end
integrated-gradients **attribution**; and generative **augmentation** (synthetic
donors and cells).

The numerical model is the `m3-sc` PyTorch engine, bundled under `inst/python/m3` and driven via **reticulate** inside a **basilisk** environment. The R package only modernises the call side (objects in, objects
out) — results are **identical to the Python package** for a given seed, device
and library versions.

## Install

```r
# install.packages("remotes")
remotes::install_github("PYangLab/M3", subdir = "m3-r")
```

The first `m3_train()` provisions a private Python environment via `basilisk`
(torch + scanpy + captum, CPU or GPU). No system Python is required.

## Quickstart

```r
library(m3)

data  <- m3_demo()                       # built-in Liu CITE-seq demo (3 batches)
model <- m3_train(
  data,
  condition_keys   = c("cond_group", "Age_interval"),
  target_condition = "cond_group",
  celltype_key     = "mergedcelltype",
  donor_key        = "sample_id",
  batch_key        = "batch",
  held_out         = "B3",           # leak-safe leave-one-batch-out
  embedding_dim    = 30L,
  max_epochs       = 500L,
  seed             = 0L
)

emb   <- m3_embedding(model, part = "bio")      # cells x latent (batch removed)
preds <- m3_predict_donors(model)               # per-donor disease probabilities
attr  <- m3_attribute(model, reference_labels = "HC")
m3_top_genes(attr, n = 100, modality = "rna")
m3_top_celltypes(attr, min_cells_per_condition = 200)
aug   <- m3_augment(model, conditions = c("HC","Severe"), n_donors = c(3,3), tau = 0.8)
```

Build a dataset from your own data (a `SingleCellExperiment` with RNA in the main
assay and ADT/ATAC as `altExp`s; role columns in `colData`):

```r
data <- m3_dataset(sce, batch = "batch1")
# or read paper-format HDF5 directly (big data stays in Python):
data <- m3_concat(list(
  m3_read_h5(rna = "rna1.h5", adt = "adt1.h5", metadata = "metadata1.csv", batch = "batch1"),
  m3_read_h5(rna = "rna2.h5", adt = "adt2.h5", metadata = "metadata2.csv", batch = "batch2")
))
```

## Function map (R ↔ Python)

| R | Python (`import m3`) |
|---|---|
| `m3_dataset()` / `m3_read_h5()` / `m3_concat()` / `m3_demo()` | `m3.read_matrix` / `m3.read_h5` / `m3.concat` / `m3.datasets.liu_demo` |
| `m3_train()` | `m3.M3(...).train(...)` |
| `m3_embedding()` / `m3_reconstruct()` / `m3_cell_metadata()` | `M3.embedding` / `reconstruct` / `cell_metadata` |
| `m3_predict_donors()` / `m3_donor_embedding()` | `M3.predict_donors` / `donor_embedding` |
| `m3_attribute()` → `m3_top_genes()` / `m3_top_celltypes()` | `M3.attribute(...)` → `Attribution.top_genes` / `top_celltypes` |
| `m3_augment()` / `m3_generate()` | `M3.augment` / `generate` |
| `m3_umap()` / `m3_pca2()` | scanpy/umap-learn/sklearn (tutorial projections) |

## Reproducibility / parity

`m3_train(seed = )` seeds the Stage-1 integration VAE, which the upstream engine
leaves **unseeded**. With a fixed seed, device and library versions, R and Python
produce identical results.

The tutorials (`inst/tutorials/`, demo + full-data variants of all four) compute
their UMAP/PCA projections through `m3_umap()`/`m3_pca2()` — the same
scanpy/umap-learn code the Python tutorials use — so even the projection
**coordinates** match (no uwot-vs-umap-learn rotation).

## Status

Work in progress: the call-side R wrapper of the published m3 engine. The
numerical model is frozen.
