---
tags:
  - reference
  - api
---

# R API

The R package is imported with `library(m3)`. It has a one-to-one counterpart in
the [Python API](api-python.md) — both drive the same vendored engine, so a
script translated verb-for-verb produces identical results.

R objects (`m3_dataset`, `m3_model`, `m3_attribution`) are light **handles** to
live objects in a Python worker. The worker is started on first use by
[**basilisk**](https://bioconductor.org/packages/basilisk/), which provisions the
engine automatically — there is nothing to `pip install` yourself.

**Conventions shared by every verb:**

- **Condition-aware.** `condition_keys` is required — m3 always models at least
  one condition factor.
- **Handles are session-bound.** `m3_dataset` / `m3_model` / `m3_attribution`
  live in the worker; call [`m3_shutdown()`](#m3_shutdown) to release them, and
  re-train to continue.
- **Device.** Every entry point takes `device = c("auto", "cpu", "cuda")`, fixed
  on the first worker start; changing it needs `m3_shutdown()` then a fresh call.
- **Attach-and-carry.** Every read-out verb accepts either an `m3_model` or a
  `SingleCellExperiment` / `MultiAssayExperiment` that carries a model attached
  with [`m3_attach()`](#m3_attach) (the Seurat-style flow).
- **Leak-safe holdout.** `held_out=` (whole batches) or `held_out_samples=`
  (specific donors) masks the query set's `target_condition` during training, so
  [`m3_predict_donors()`](#m3_predict_donors) reports honest generalisation.

A typical run is a handful of lines:

```r
library(m3)

data  <- m3_demo()                                   # 1. load a dataset
model <- m3_train(                                   # 2. declare column roles + train
  data,
  condition_keys = "cond_group",
  celltype_key   = "mergedcelltype",
  donor_key      = "sample_id"
)
emb   <- m3_embedding(model, part = "bio")           # 3. read out
```

## Capability gating

Which read-outs are available depends on how the model was trained. Every trained
model unlocks the integration-level read-outs
([`m3_embedding()`](#m3_embedding), [`m3_reconstruct()`](#m3_reconstruct),
[`m3_generate()`](#m3_generate), [`m3_augment()`](#m3_augment)). The donor-level
read-outs ([`m3_predict_donors()`](#m3_predict_donors),
[`m3_donor_embedding()`](#m3_donor_embedding), [`m3_attribute()`](#m3_attribute))
additionally require the **donor predictor**, which trains automatically when both
`donor_key` and `celltype_key` are supplied (and is skipped with
`donor_prediction = FALSE`). Inspect with [`m3_capabilities()`](#m3_capabilities):

```r
m3_capabilities(model)
#> embedding reconstruct predict_donors
#>      TRUE        TRUE           TRUE
```

---

## Data & inputs

All readers return an `m3_dataset` holding **raw counts** — m3 normalises
internally, so do **not** pre–log-normalise the input.

| Function | Description |
|---|---|
| [`m3_demo()`](#m3_demo) | load the built-in Liu *et al.* CITE-seq demo dataset |
| [`m3_read_h5()`](#m3_read_h5) | read one batch from paper-format HDF5 matrices + a metadata CSV |
| [`m3_read_h5ad()`](#m3_read_h5ad) | read one batch from an AnnData `.h5ad` |
| [`m3_dataset()`](#m3_dataset) | build a dataset from an SCE / MAE / Seurat / list |
| [`m3_concat()`](#m3_concat) | combine single-batch datasets into one multi-batch dataset |
| [`m3_example_sce()`](#m3_example_sce) | a small synthetic RNA + ADT `SingleCellExperiment` for examples |
| [`m3_dataset_obs()`](#m3_dataset_obs) | per-cell metadata (obs) of a dataset |
| [`m3_dataset_matrix()`](#m3_dataset_matrix) | dense count matrix for one modality of a dataset |

### `m3_demo()`

```r
m3_demo(device = c("auto", "cpu", "cuda"))
```

Load the built-in **Liu et al.** COVID-19 demo — the same stratified subsample
(RNA HVGs + ADT, 3 batches `B1`/`B2`/`B3`) the Python package ships, so demo
results match exactly.

**Returns:** an `m3_dataset`.

Maps to Python [`m3.datasets.liu_demo()`](api-python.md#m3datasetsliu_demo).

### `m3_read_h5()`

```r
m3_read_h5(rna = NULL, adt = NULL, atac = NULL, metadata,
           batch = "batch0", device = c("auto", "cpu", "cuda"))
```

Read one batch from paper-format HDF5 count matrices (counts under `matrix/data`,
names under `matrix/features`) plus a per-cell metadata CSV. Big data stays in
Python.

| Parameter | Meaning |
|---|---|
| `rna` | path to the RNA `.h5` file. |
| `adt` | path to the ADT `.h5` file (optional). |
| `atac` | path to the ATAC `.h5` file (optional). |
| `metadata` | path to the per-cell metadata `.csv` providing the role columns *(required)*. |
| `batch` | batch label written to `obs$batch` (default `"batch0"`). |
| `device` | session device (default `"auto"`). |

**Returns:** an `m3_dataset`.

Maps to Python [`m3.read_h5()`](api-python.md#m3read_h5).

### `m3_read_h5ad()`

```r
m3_read_h5ad(path, batch = "batch0", modality = "rna",
             device = c("auto", "cpu", "cuda"))
```

Read one batch from an AnnData `.h5ad` file. If `var` has a `feature_types`
column the matrix is split into modalities; otherwise the whole matrix is taken
as `modality`.

| Parameter | Meaning |
|---|---|
| `path` | path to the `.h5ad` file. |
| `batch` | batch label (default `"batch0"`). |
| `modality` | modality name used when `var` has no `feature_types` column (default `"rna"`). |
| `device` | session device (default `"auto"`). |

**Returns:** an `m3_dataset`.

Maps to Python [`m3.read_h5ad()`](api-python.md#m3read_h5ad).

### `m3_dataset()`

```r
m3_dataset(x, batch = "batch0", assay = "counts", adt_exp = "ADT",
           atac_exp = "ATAC", device = c("auto", "cpu", "cuda"))
```

Build an `m3_dataset` from an in-memory object: RNA raw counts in the main assay,
ADT/ATAC as `altExps`, with the role columns in `colData`. `m3_train()` calls
this for you, so you can also pass the object straight to `m3_train()`.

| Parameter | Meaning |
|---|---|
| `x` | a `SingleCellExperiment`, `MultiAssayExperiment`, `Seurat` object, or a list with `rna`/`adt`/`atac` (features × cells) plus `obs`. |
| `batch` | batch label written to `obs$batch` (default `"batch0"`). |
| `assay` | assay selector for the RNA main assay of SCE input (default `"counts"`). |
| `adt_exp` | `altExp` selector for ADT in SCE input (default `"ADT"`). |
| `atac_exp` | `altExp` selector for ATAC in SCE input (default `"ATAC"`). |
| `device` | session device (default `"auto"`). |

**Returns:** an `m3_dataset`.

Maps to Python [`m3.from_anndata()`](api-python.md#m3from_anndata) /
[`m3.read_matrix()`](api-python.md#m3read_matrix).

### `m3_concat()`

```r
m3_concat(datasets)
```

Combine single-batch datasets into one multi-batch dataset. Features per modality
are harmonised by **name intersection** (the first batch's order is kept); batch
labels must be unique across inputs.

| Parameter | Meaning |
|---|---|
| `datasets` | a list of `m3_dataset` objects. |

**Returns:** an `m3_dataset`.

Maps to Python [`m3.concat()`](api-python.md#m3concat).

### `m3_example_sce()`

```r
m3_example_sce(n_cells = 180L, seed = 1L)
```

Build a small synthetic RNA + ADT `SingleCellExperiment` with the role columns m3
expects (`mergedcelltype`, `cond_group` `HC`/`Severe`, `Age_interval`,
`sample_id`, `batch` `B1`/`B2`) — for examples and quick CPU trials.

| Parameter | Meaning |
|---|---|
| `n_cells` | number of cells (default `180`). |
| `seed` | RNG seed for the synthetic counts/labels (default `1`). |

**Returns:** a `SingleCellExperiment` with an `"ADT"` `altExp` and role columns in
`colData`. *(R-only convenience; no Python counterpart.)*

### `m3_dataset_obs()`

```r
m3_dataset_obs(dataset)
```

Per-cell metadata (`obs`) of a dataset as a `data.frame`, one row per cell.

**Returns:** a `data.frame`.

Maps to Python `Dataset.obs`.

### `m3_dataset_matrix()`

```r
m3_dataset_matrix(dataset, modality)
```

Dense count matrix (cells × features) for one `modality` of a dataset (e.g.
`"rna"`).

**Returns:** a numeric matrix, cells × features.

Maps to Python `Dataset.modalities[...]`.

---

## Model & training

| Function | Description |
|---|---|
| [`m3_train()`](#m3_train) | train the integration VAE (+ optional donor predictor) |
| [`m3_attach()`](#m3_attach) | store a trained model inside an SCE / MAE |
| [`m3_capabilities()`](#m3_capabilities) | which read-outs a model unlocks |
| [`m3_reference_vocab()`](#m3_reference_vocab) | leak-safe reference vocabulary the model trained on |

### `m3_train()`

```r
m3_train(data, condition_keys, target_condition = NULL, celltype_key = NULL,
         donor_key = NULL, batch_key = NULL, held_out = NULL,
         held_out_samples = NULL, hvg = NULL, embedding_dim = 30L,
         max_epochs = 300L, lr = 1e-5, batch_size = 256L,
         early_stop_patience = 300L, min_delta = 0, val_percentage = 0.1,
         weight_batch_ae = 1, weight_modality = NULL, balance_batches = TRUE,
         donor_prediction = NULL, donor_predictor = NULL, seed = 0L,
         device = c("auto", "cpu", "cuda"))
```

Train the Stage-1 integration VAE and — when `donor_key` + `celltype_key` are
given — the Stage-2 adversarial donor-level disease predictor on top. With
`held_out` / `held_out_samples` the query set is leak-safe (its
`target_condition` labels are masked, then predicted).

| Parameter | Meaning |
|---|---|
| `data` | an `m3_dataset`, or any input `m3_dataset()` accepts (SCE / MAE / Seurat / list), converted automatically. |
| `condition_keys` | character vector of condition columns in `obs` *(required)*; each gets its own latent factor. |
| `target_condition` | the condition to predict / attribute (default: the first of `condition_keys`). |
| `celltype_key` | cell-type role column; the donor predictor + attribution need `donor_key` + `celltype_key` (default `NULL`). |
| `donor_key` | donor/sample role column; needed by the donor predictor + attribution (default `NULL`). |
| `batch_key` | batch role column (effective default `"batch"`); the held-out / integration unit and the site the donor adversary removes. |
| `held_out` | character vector of batch labels to hold out as an unlabelled query set (default `NULL`). |
| `held_out_samples` | character vector of donor IDs to hold out as the query set, possibly spanning batches; mutually exclusive with `held_out` (default `NULL`). |
| `hvg` | optional named list of per-modality HVG counts, e.g. `list(rna = 1000)` (default `NULL` = all features). |
| `embedding_dim` | latent width (default `30`). |
| `max_epochs` | Stage-1 max training epochs (default `300`). |
| `lr` | Stage-1 learning rate (default `1e-5`). |
| `batch_size` | Stage-1 training batch size (default `256`). |
| `early_stop_patience` | Stage-1 early-stopping patience in epochs (default `300`). |
| `min_delta` | Stage-1 early-stopping minimum improvement (default `0`). |
| `val_percentage` | fraction held for validation during Stage-1 (default `0.1`). |
| `weight_batch_ae` | Stage-1 batch-autoencoder loss weight (default `1`). |
| `weight_modality` | optional per-modality reconstruction-loss weights (named list or vector); `NULL` weights every present modality equally. |
| `balance_batches` | train the VAE on a batch-balanced subset (default `TRUE`). |
| `donor_prediction` | force the donor predictor on/off; `NULL` (default) enables it when `donor_key` + `celltype_key` are present. |
| `donor_predictor` | named list of donor-predictor knobs, e.g. `list(glr = 3e-3, n_epochs = 120, min_cells = 5)`; `NULL` uses the driver defaults. |
| `seed` | integer seed applied before training so the otherwise-unseeded Stage-1 VAE is reproducible (default `0`; `NULL` keeps unseeded behaviour). |
| `device` | one of `"auto"`, `"cpu"`, `"cuda"`, fixed on first use (default `"auto"`). |

**Returns:** an `m3_model`.

Maps to Python [`m3.M3(...).train()`](api-python.md#m3m3) — the R call fuses the
Python constructor and `train()` into one step.

### `m3_attach()`

```r
m3_attach(object, model)
```

Store a trained `model` inside a `SingleCellExperiment` / `MultiAssayExperiment`
(`metadata(object)$m3`) so the object carries its model into the read-out verbs
(the Seurat-style flow).

**Returns:** `object` with the model attached. *(R-only convenience; the Python
`M3` object already holds its own state.)*

### `m3_capabilities()`

```r
m3_capabilities(model)
```

Capabilities of a trained model (`embedding` / `reconstruct` / `predict_donors`).

**Returns:** a named logical vector.

Maps to Python `M3.capabilities`.

### `m3_reference_vocab()`

```r
m3_reference_vocab(model)
```

The leak-safe reference vocabulary the model trained on (query labels excluded) —
one entry per condition key.

**Returns:** a named list.

Maps to Python `M3.contract["reference_vocab"]`.

---

## Read-outs

| Function | Description |
|---|---|
| [`m3_embedding()`](#m3_embedding) | cell-level latent embedding, by latent part |
| [`m3_cell_metadata()`](#m3_cell_metadata) | row-aligned cell metadata for the read-outs |
| [`m3_reconstruct()`](#m3_reconstruct) | batch-corrected per-modality reconstruction |
| [`m3_predict_donors()`](#m3_predict_donors) | donor-level disease prediction |
| [`m3_donor_embedding()`](#m3_donor_embedding) | patient/donor-level embedding |
| [`m3_generate()`](#m3_generate) | posterior-resampled cells (1:1) |
| [`m3_augment()`](#m3_augment) | synthesise new donors per condition |

### `m3_embedding()`

```r
m3_embedding(model, part = "bio")
```

Cell-level latent embedding, row-aligned to [`m3_cell_metadata()`](#m3_cell_metadata).
`part` selects the factor(s): `"bio"` (intrinsic + condition latents, batch
removed — the integration embedding for clustering / UMAP), `"intrinsic"`,
`"batch"`, or any of the model's condition keys.

**Returns:** a numeric matrix, cells × latent dimensions.

Maps to Python [`M3.embedding()`](api-python.md#m3embedding).

### `m3_cell_metadata()`

```r
m3_cell_metadata(model)
```

Row-aligned cell metadata for the embedding / reconstruction rows (reference rows
first, then any held-out query rows).

**Returns:** a `data.frame`.

Maps to Python `M3.cell_metadata`.

### `m3_reconstruct()`

```r
m3_reconstruct(model, remove_batch = TRUE)
```

Batch-corrected per-modality reconstruction (posterior-mean decode). With
`remove_batch = TRUE` (default) the batch latent is zeroed before decoding,
yielding a batch-corrected matrix suitable for cross-batch differential
expression.

**Returns:** a named list of numeric matrices (one per modality), cells ×
features.

Maps to Python [`M3.reconstruct()`](api-python.md#m3reconstruct).

### `m3_predict_donors()`

```r
m3_predict_donors(model, include_reference = FALSE)
```

Donor-level prediction of `target_condition`: per-donor predicted label + class
probabilities. Requires the donor predictor. By default only the held-out query
donors are returned; pass `include_reference = TRUE` to include the reference
donors too.

**Returns:** a `data.frame`: `donor`, `is_reference`, `predicted_label`, and one
`prob_<label>` column per class.

Maps to Python [`M3.predict_donors()`](api-python.md#m3predict_donors).

### `m3_donor_embedding()`

```r
m3_donor_embedding(model)
```

The corrected per-donor vectors that get classified — a patient-level embedding.
Requires the donor predictor.

**Returns:** a `data.frame` with a `donor` column, an `is_reference` flag, and the
embedding dimensions (`m3_0`, `m3_1`, …).

Maps to Python [`M3.donor_embedding()`](api-python.md#m3donor_embedding).

### `m3_generate()`

```r
m3_generate(model, tau = 0.8, seed = 42L)
```

Posterior-resampled synthetic cells (1:1 with the reference cells). `tau` scales
the sampling temperature.

**Returns:** a named list of numeric matrices (one per modality), cells ×
features.

Maps to Python [`M3.generate()`](api-python.md#m3generate).

### `m3_augment()`

```r
m3_augment(model, conditions, n_donors, tau = 0.8, batch = NULL, seed = 42L)
```

Synthesise **new donors** per condition by resampling real donor templates
through the VAE posterior. Requires `donor_key` + `celltype_key`.

| Parameter | Meaning |
|---|---|
| `conditions` | character vector of conditions to synthesise (in order). |
| `n_donors` | integer vector, synthetic donors per condition (same length as `conditions`). |
| `tau` | posterior temperature (default `0.8`). |
| `batch` | optional batch label to template from (needs `batch_key` at training); `NULL` draws from all batches. |
| `seed` | integer seed (default `42`). |

**Returns:** a list with `expression` (named list of cells × features matrices,
one per modality) and `obs` (a `data.frame`, one row per synthetic cell).

Maps to Python [`M3.augment()`](api-python.md#m3augment).

---

## Attribution

| Function | Description |
|---|---|
| [`m3_attribute()`](#m3_attribute) | integrated-gradients attribution of the donor prediction |
| [`m3_top_genes()`](#m3_top_genes) | per-cell-type-balanced top genes (the publication recipe) |
| [`m3_top_celltypes()`](#m3_top_celltypes) | cell-type importance ranking |
| [`m3_attribution_matrix()`](#m3_attribution_matrix) | full cell × gene signed attribution matrix |
| [`m3_gene_celltype_matrix()`](#m3_gene_celltype_matrix) | signed per-(cell-type, gene) attribution matrix |

### `m3_attribute()`

```r
m3_attribute(model, reference_labels, target_class = NULL, n_steps = 50L)
```

End-to-end integrated-gradients attribution of the donor-level prediction back to
genes/proteins, cell types and donors. Requires the donor predictor.

| Parameter | Meaning |
|---|---|
| `reference_labels` | baseline label(s) of `target_condition` (e.g. `"HC"`); without them the engine falls back to a zero baseline that inflates housekeeping genes. |
| `target_class` | optional explicit target class index (default: the first non-reference class). |
| `n_steps` | integrated-gradients steps (default `50`). |

**Returns:** an `m3_attribution` with ranked tables `$genes`, `$celltypes`,
`$donors`; pass it to `m3_top_genes()` / `m3_top_celltypes()` /
`m3_attribution_matrix()`.

Maps to Python [`M3.attribute()`](api-python.md#m3attribute).

### `m3_top_genes()`

```r
m3_top_genes(attribution, n = 100L, min_cells_per_condition = 200L,
             exclude_regex, modality = NULL)
```

The publication ranking recipe: drops sparse cell types, scores each gene by
`mean(|gene × celltype IG|)`, excludes housekeeping / ribosomal genes, and
optionally restricts to one modality.

| Parameter | Meaning |
|---|---|
| `attribution` | an `m3_attribution` from `m3_attribute()`. |
| `n` | number of genes to return (default `100`). |
| `min_cells_per_condition` | cell-type filter threshold (default `200`; `0` to skip). |
| `exclude_regex` | regex of names to drop; **omitted** = default housekeeping pattern, `NULL` = no exclusion, a string = that regex. |
| `modality` | `"rna"`/`"adt"`/`"atac"` to restrict ranking; `NULL` for all. |

**Returns:** a `data.frame`: `feature`, `modality`, `score`, `n_celltypes_used`.

Maps to Python [`Attribution.top_genes()`](api-python.md#attributiontop_genes).

### `m3_top_celltypes()`

```r
m3_top_celltypes(attribution, min_cells_per_condition = 200L)
```

Cell-type importance ranking, filtered to types with at least
`min_cells_per_condition` cells in **both** conditions (default `200`; `0` returns
the raw table).

**Returns:** a `data.frame`: `celltype`, `importance`.

Maps to Python [`Attribution.top_celltypes()`](api-python.md#attributiontop_celltypes).

### `m3_attribution_matrix()`

```r
m3_attribution_matrix(attribution)
```

The full cell × gene signed attribution matrix.

**Returns:** a numeric matrix, cells × features.

Maps to Python `Attribution.attribution`.

### `m3_gene_celltype_matrix()`

```r
m3_gene_celltype_matrix(attribution)
```

The signed per-(cell-type, gene) attribution matrix.

**Returns:** a numeric matrix, cell types × features (rows =
`attribution$celltype_names`).

Maps to Python `Attribution.gene_celltype_matrix`.

---

## Utilities

| Function | Description |
|---|---|
| [`m3_umap()`](#m3_umap) | UMAP projection identical to the Python tutorials |
| [`m3_pca2()`](#m3_pca2) | two-component PCA (small-sample fallback) |
| [`m3_shutdown()`](#m3_shutdown) | release the Python engine session |

### `m3_umap()`

```r
m3_umap(x, method = c("scanpy", "umap"), n_neighbors = 15L, min_dist = 0.1,
        spread = 1.0, metric = "euclidean", random_state = 0L, device = "auto")
```

UMAP projection identical to the m3 Python tutorials — runs scanpy
`neighbors`+`umap` (method `"scanpy"`) or `umap.UMAP` (method `"umap"`) inside the
engine environment so coordinates match Python exactly.

| Parameter | Meaning |
|---|---|
| `x` | a numeric matrix (rows = points), e.g. an `m3_embedding()`. |
| `method` | `"scanpy"` or `"umap"` (default `"scanpy"`). |
| `n_neighbors` | UMAP neighbours (default `15`; capped at `nrow - 1`). |
| `min_dist`, `spread`, `metric`, `random_state` | umap-learn parameters for method `"umap"`. |
| `device` | session device (default `"auto"`; pass the model's device). |

**Returns:** a 2-column numeric matrix of coordinates.

*(The Python tutorials call scanpy directly; `m3_umap()` wraps the same routine
so R and Python UMAPs coincide.)*

### `m3_pca2()`

```r
m3_pca2(x, device = "auto")
```

Two-component PCA (scikit-learn) — the small-sample fallback used in Tutorial 4.

**Returns:** a 2-column numeric matrix.

### `m3_shutdown()`

```r
m3_shutdown()
```

Shut down the m3 Python session — releases the basilisk process and every live
model held in it (models are session-bound; re-train to continue).

**Returns:** invisible `NULL`. *(R-only — Python has no separate session to
release.)*

---

## Object classes

The handles returned above are S3 objects with `print()` methods:

- **`m3_dataset`** — a live dataset (`print()` shows cell count, batches,
  modalities).
- **`m3_model`** — a trained model (`print()` shows modalities, `embedding_dim`,
  condition keys + target, role columns, held-out set, capabilities).
- **`m3_attribution`** — an attribution result (`print()` shows the target label
  and the row counts of `$genes` / `$celltypes` / `$donors`).

All model-consuming verbs accept either an `m3_model` or an SCE/MAE with a model
attached via [`m3_attach()`](#m3_attach).

---

## Python ⇄ R correspondence

| R | Python |
|---|---|
| `m3_demo()` | `m3.datasets.liu_demo()` |
| `m3_read_h5()` / `m3_read_h5ad()` | `m3.read_h5()` / `m3.read_h5ad()` |
| `m3_dataset()` | `m3.from_anndata()` / `m3.read_matrix()` |
| `m3_concat()` | `m3.concat()` |
| `m3_train()` | `m3.M3(...).train()` |
| `m3_capabilities()` | `M3.capabilities` |
| `m3_embedding()` | `M3.embedding()` |
| `m3_cell_metadata()` | `M3.cell_metadata` |
| `m3_reconstruct()` | `M3.reconstruct()` |
| `m3_predict_donors()` | `M3.predict_donors()` |
| `m3_donor_embedding()` | `M3.donor_embedding()` |
| `m3_generate()` | `M3.generate()` |
| `m3_augment()` | `M3.augment()` |
| `m3_attribute()` | `M3.attribute()` |
| `m3_top_genes()` / `m3_top_celltypes()` | `Attribution.top_genes()` / `.top_celltypes()` |
| `m3_shutdown()` | — *(R only — releases the engine session)* |

## See also

- **[Python API](api-python.md)** — the same surface for Python (`import m3`).
- **Tutorials** — [Representation learning](notebooks/r_01_representation_learning.md),
  [Patient prediction](notebooks/r_02_patient_prediction.md),
  [Feature attribution](notebooks/r_03_attribution.md),
  [Data augmentation](notebooks/r_04_augmentation.md).
