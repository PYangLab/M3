---
tags:
  - reference
  - api
---

# Python API

The package is installed from GitHub — a PyPI release as **`m3-sc`** is planned — and imported as **`import m3`**.
It has a one-to-one counterpart in the [R API](api-r.md) — both drive the same
engine and produce identical results.

The public API has two pieces:

1. an **input facade** (`m3.read_h5`, `m3.read_h5ad`, `m3.from_anndata`,
   `m3.read_matrix`, `m3.concat`, and `m3.datasets.*`) that turns common
   single-cell formats into one canonical [`Dataset`](#the-dataset-container); and
2. the [`m3.M3`](#m3m3) model object, whose methods cover representation
   learning, donor-level disease prediction, integrated-gradients attribution,
   and generative augmentation.

A typical run is three lines:

```python
import m3

data  = m3.datasets.liu_demo()                          # 1. load a Dataset
model = m3.M3(data, condition_keys=["cond_group"],      # 2. declare column roles
              celltype_key="mergedcelltype",
              donor_key="sample_id").train()    # 3. train
emb   = model.embedding(part="bio")                     #    read out
```

## Capability gating

`m3` is staged, and which read-outs are available depends on how the model was
built and trained. Every trained model unlocks the integration-level
read-outs ([`embedding`](#m3embedding), [`reconstruct`](#m3reconstruct),
[`generate`](#m3generate), [`augment`](#m3augment)). The donor-level read-outs
([`predict_donors`](#m3predict_donors), [`donor_embedding`](#m3donor_embedding),
[`attribute`](#m3attribute)) additionally require the **donor predictor**, which
trains automatically when both `donor_key` and `celltype_key` were supplied at
construction (and is skipped when you pass `donor_prediction=False`). Calling a
read-out before its capability is unlocked raises
[`m3.M3CapabilityError`](#exceptions). Inspect `model.capabilities` at any time:

```python
model.capabilities
# {'embedding': True, 'reconstruct': True, 'predict_donors': True}
```

---

## Input facade

All readers return a [`Dataset`](#the-dataset-container) holding **raw counts**.
m3 normalises internally, so do **not** pre–log-normalise the input.

### `m3.read_h5`

```python
m3.read_h5(*, rna: str | None = None, adt: str | None = None,
           atac: str | None = None, metadata: str,
           batch: str = "batch0") -> Dataset
```

Read one batch from paper-format HDF5 count matrices plus a metadata CSV. Pass
the path to each modality you have (at least one of `rna` / `adt` / `atac`) and
the per-cell `metadata` CSV. Matrix orientation (cells×features vs.
features×cells) is resolved automatically against the feature count.

### `m3.read_h5ad`

```python
m3.read_h5ad(path: str, *, batch: str = "batch0",
             modality: str = "rna") -> Dataset
```

Read one batch from an AnnData `.h5ad`. If `adata.var` has a `feature_types`
column (`Gene Expression` / `Antibody Capture` / `Peaks`), the matrix is split
into modalities; otherwise the whole matrix is taken as `modality`.

### `m3.from_anndata`

```python
m3.from_anndata(adata, *, batch: str = "batch0",
                modality: str = "rna") -> Dataset
```

Build one batch from an in-memory `AnnData`. Splitting by `feature_types`
behaves as in [`read_h5ad`](#m3read_h5ad).

### `m3.read_matrix`

```python
m3.read_matrix(*, counts: dict, obs: 'pd.DataFrame', var: dict,
               batch: str = "batch0") -> Dataset
```

Build one batch from in-memory per-modality count arrays.

- `counts` — `{modality -> array | sparse}` of raw counts, shape `[n_cells, n_features]`
- `obs` — per-cell `DataFrame`, `n_cells` rows
- `var` — `{modality -> list/Index}` of feature names, aligned to each matrix's columns

### `m3.concat`

```python
m3.concat(datasets: 'list[Dataset]') -> Dataset
```

Stack single-batch `Dataset`s into one multi-batch `Dataset`. Features per
modality are harmonised by **name intersection** (the first batch's order is
kept); a batch missing a whole modality is zero-filled and flagged in
`present` (mosaic design). Batch labels must be unique across inputs.

### `m3.datasets.liu_demo`

```python
m3.datasets.liu_demo() -> Dataset
```

Load the built-in **Liu et al.** COVID-19 demo — a **~30 k-cell**
(30,534) stratified subsample of the full
~135 k-cell dataset, shipped inside the wheel so every tutorial runs with no
downloads.

```python
>>> data = m3.datasets.liu_demo()
>>> print(data)
Dataset(n_cells=30534, batches=['B1', 'B2', 'B3'], modalities=[rna:1000, adt:192])
```

`obs` carries the columns `batch` (`B1` / `B2` / `B3`), `sample_id`, `Donor`,
`cond_group` (`HC` / `Severe`), `Age_interval`, and `mergedcelltype`.

---

## The `Dataset` container

The readers return an immutable, multimodal, multi-batch **raw-count**
container. You normally obtain one from a reader rather than constructing it by
hand. Useful read-only members:

| Member | Meaning |
|---|---|
| `data.modalities` | `{name -> scipy.sparse.csr_matrix}` of raw counts |
| `data.obs` | per-cell `DataFrame` (always contains a `batch` column) |
| `data.var` | `{name -> pd.Index}` of feature names per modality |
| `data.present` | `{name -> bool ndarray}` per-cell modality presence (mosaic) |
| `data.n_cells` | number of cells |
| `data.modality_names` | list of modality names, e.g. `['rna', 'adt']` |
| `data.batches` | list of batch labels |

```python
>>> data.modality_names
['rna', 'adt']
>>> data.obs["mergedcelltype"].value_counts().head(3)
```

---

## `m3.M3`

```python
m3.M3(
    dataset,
    *,
    condition_keys,                  # str | list[str]  (required)
    target_condition: str | None = None,
    donor_key: str | None = None,
    celltype_key: str | None = None,
    batch_key: str = "batch",
    held_out: list | None = None,
    held_out_samples: list | None = None,
    hvg: dict | None = None,
    embedding_dim: int = 30,
)
```

Construct a model and **freeze the column-role contract** against `dataset.obs`.
Nothing is trained yet — call [`train`](#m3train).

| Argument | Role |
|---|---|
| `condition_keys` | one or more `obs` columns the model is made aware of; each gets its own latent factor. m3 is condition-aware, so this is required. |
| `target_condition` | the single condition used for prediction / attribution (defaults to the first of `condition_keys`). |
| `donor_key` | `obs` column identifying a donor/sample. Required for donor-level prediction, attribution, and `augment`. |
| `celltype_key` | `obs` column with cell-type labels. Required for [`train`](#m3train). |
| `batch_key` | `obs` column with the batch label (default `"batch"`, written by every reader). It is the integration / held-out unit, the *site* the donor adversary removes, and the stratifier for [`augment(batch=...)`](#m3augment). |
| `held_out` | batch label(s) to hold out as an unlabelled **query** set — their `target_condition` labels are masked during training and recovered by [`predict_donors`](#m3predict_donors). |
| `held_out_samples` | donor/sample ID(s) to hold out as the **query** set instead of a whole batch — may span batches; requires `donor_key` and is mutually exclusive with `held_out`. Their `target_condition` labels are masked during training and recovered by [`predict_donors`](#m3predict_donors). |
| `hvg` | `{modality -> n_top}` highly-variable-feature selection, e.g. `{"rna": 1000}`. Omit to use all features as given. |
| `embedding_dim` | size of the shared biological latent (default `30`). |

After construction, `model.contract` records the frozen roles, including
`model.contract["reference_vocab"]` (the condition values observed in the
reference batches).

### `M3.train`

```python
M3.train(
    *,
    max_epochs: int = 300,
    lr: float = 1e-5,
    batch_size: int = 256,
    early_stop_patience: int = 300,
    min_delta: float = 0.0,
    val_percentage: float = 0.1,
    weight_batch_ae: float = 1.0,
    balance_batches: bool = True,
    donor_prediction: bool | None = None,
    donor_predictor: dict | None = None,
) -> M3
```

Train the integration VAE and, when applicable, the donor predictor on top.
Returns the model (so you can chain `.train()`). Uses CUDA automatically when
available; otherwise CPU.

- **`donor_prediction`** — `None` (default) auto-enables the donor predictor
  when `donor_key` and `celltype_key` are set. Pass `False` for pure
  representation learning, `True` to require it.
- **`balance_batches`** — when `True` (default), the VAE trains on a
  batch-balanced subset; with no held-out query, downstream steps are
  automatically rebuilt on the **full** reference so sparse cell types are not
  biased out.
- **`donor_predictor`** — a dict tuning the Stage-2 adversarial corrector.
  Recognised keys (defaults): `enc_lr` (`1e-5`), `glr` (`5e-3`), `n_epochs`
  (`30`), `patient_w` (`3.0`), `adv_max` (`1.0`), `adv_warmup` (`10`), `n_disc`
  (`1`), `hidden` (`64`), `min_cells` (`5`).

---

## Read-outs

### `m3.embedding`

```python
M3.embedding(part: str = "bio") -> np.ndarray
```

Cell-level latent, row-aligned to [`cell_metadata`](#m3cell_metadata).
`part` selects which factor(s):

- `"bio"` — the intrinsic biological latent plus every condition factor (the
  integration embedding for clustering / UMAP)
- `"intrinsic"` — the intrinsic biological latent only
- `"batch"` — the 2-d batch factor (should carry batch signal, not biology)
- any name in `condition_keys` — that condition's 2-d factor

### `m3.reconstruct`

```python
M3.reconstruct(*, remove_batch: bool = True) -> dict
```

Posterior-mean decode of every cell, returned as `{modality -> ndarray}`
(`[n_cells, n_features]`). With `remove_batch=True` (default) the batch factor is
zeroed before decoding, yielding a batch-corrected matrix suitable for
cross-batch differential expression.

### `m3.predict_donors`

```python
M3.predict_donors(*, include_reference: bool = False) -> pd.DataFrame
```

Donor-level prediction of `target_condition`. Requires the donor predictor.
Returns one row per donor with columns `donor`, `is_reference`,
`predicted_label`, and a `prob_<label>` column per class. By default only the
held-out **query** donors are returned; pass `include_reference=True` to include
the reference donors too.

### `m3.donor_embedding`

```python
M3.donor_embedding() -> pd.DataFrame
```

The corrector-corrected per-donor vector used for prediction — a `DataFrame`
indexed by donor, with an `is_reference` column followed by embedding dimensions
(`m3_0`, `m3_1`, …). Join with `predict_donors(include_reference=True)` for
labels, or feed straight into UMAP for a patient-level map. Requires the donor
predictor.

### `m3.generate`

```python
M3.generate(*, tau: float = 0.8, seed: int = 42) -> dict
```

Posterior-resampled synthetic cells (one per reference cell, 1:1), returned as
`{modality -> ndarray}`. `tau` scales the sampling temperature. Handy for
noise-augmenting a training set.

### `m3.augment`

```python
M3.augment(*, conditions: list, n_donors: list, tau: float = 0.8,
           batch: str | None = None, seed: int = 42) -> dict
```

Synthesise **new donors** per condition by posterior-resampling real donor
templates. `conditions` lists the `target_condition` values to generate and
`n_donors` how many synthetic donors for each (same length). Returns
`{"expression": {modality -> ndarray}, "obs": DataFrame}`. When `batch` is given
(a value of `batch_key`), templates are drawn only from that batch — useful for
reproducing batch-stratified figures. Requires `donor_key` + `celltype_key`.

### `m3.cell_metadata`

```python
M3.cell_metadata  # property -> pd.DataFrame
```

Row-aligned metadata for [`embedding`](#m3embedding) /
[`reconstruct`](#m3reconstruct) (reference cells first, then query cells).

---

## Attribution

### `m3.attribute`

```python
M3.attribute(*, reference_labels: list, target_class: int | None = None,
             n_steps: int = 50) -> Attribution
```

End-to-end integrated-gradients attribution of `target_condition`, returning an
[`Attribution`](#the-attribution-object). Requires the donor predictor.

- **`reference_labels`** — the healthy/baseline label(s) of `target_condition`
  (e.g. `["HC"]`). These set the IG baseline at **two** levels: the mean
  expression of reference cells (cell-level IG) and the mean patient vector over
  reference donors (donor-level IG). Supplying them is what keeps housekeeping /
  pan–cell-type genes from dominating the ranking.
- **`target_class`** — which non-reference class to attribute toward; defaults to
  the first class not in `reference_labels`.
- **`n_steps`** — IG interpolation steps (default `50`; lower is faster but
  biased).

### The `Attribution` object

Returned by [`attribute`](#m3attribute). Holds several views of importance:

| Member | Type | Meaning |
|---|---|---|
| `attr.genes` | `DataFrame(feature, importance)` | raw `mean(\|IG\|)` over all cells — quick, unfiltered |
| `attr.celltypes` | `DataFrame(celltype, importance)` | per–cell-type importance |
| `attr.donors` | `DataFrame(donor, attribution)` or `None` | per-donor attribution |
| `attr.cells` | `ndarray` | signed per-cell importance |
| `attr.attribution` | `ndarray [n_cells, n_features]` | full cell × feature IG matrix |
| `attr.gene_celltype_matrix` | `ndarray [n_celltypes, n_features]` | signed per–(cell-type, feature) attribution |
| `attr.target_label` | `str` | the attributed (non-reference) condition |

#### `Attribution.top_genes`

```python
Attribution.top_genes(n: int = 100, *, min_cells_per_condition: int = 200,
                      exclude_regex: str | None = <housekeeping>,
                      modality: str | None = None) -> pd.DataFrame
```

The publication ranking recipe. Drops cell types where either the reference or
the target condition has fewer than `min_cells_per_condition` cells, scores each
feature as `mean(|gene_celltype_matrix|)` across the kept cell types, removes
housekeeping / ribosomal genes by name (`exclude_regex`, default matches
`^MT-`, `^RPL`, `^RPS`, …; pass `None` to skip), optionally restricts to one
`modality` (`"rna"` / `"adt"` / `"atac"`), and returns the top-`n` ranked
`DataFrame` with columns `feature`, `modality`, `score`, `n_celltypes_used`.

#### `Attribution.top_celltypes`

```python
Attribution.top_celltypes(*, min_cells_per_condition: int = 200) -> pd.DataFrame
```

Cell-type importance ranking, filtered to types with at least
`min_cells_per_condition` cells in **both** the reference and the target
condition (default `200`; set `0` for the raw `celltypes` table).

---

## Exceptions

- `m3.M3CapabilityError` — a read-out was requested before its capability was
  unlocked (subclasses `RuntimeError`). The message names what is missing and
  how to obtain it.

---

## Versioning

The installed version is available as `m3.__version__`.

---

## R ⇄ Python correspondence

Every verb has a 1:1 counterpart in the [R API](api-r.md). The R package wraps the
same engine, so a script translated line-for-line produces identical numbers.

| Python | R |
|---|---|
| `m3.datasets.liu_demo()` | `m3_demo()` |
| `m3.read_h5()` / `m3.read_h5ad()` | `m3_read_h5()` / `m3_read_h5ad()` |
| `m3.from_anndata()` / `m3.read_matrix()` | `m3_dataset()` |
| `m3.concat()` | `m3_concat()` |
| `m3.M3(...).train()` | `m3_train(...)` |
| `M3.capabilities` | `m3_capabilities()` |
| `M3.embedding()` | `m3_embedding()` |
| `M3.cell_metadata` | `m3_cell_metadata()` |
| `M3.reconstruct()` | `m3_reconstruct()` |
| `M3.predict_donors()` | `m3_predict_donors()` |
| `M3.donor_embedding()` | `m3_donor_embedding()` |
| `M3.generate()` | `m3_generate()` |
| `M3.augment()` | `m3_augment()` |
| `M3.attribute()` | `m3_attribute()` |
| `Attribution.top_genes()` / `.top_celltypes()` | `m3_top_genes()` / `m3_top_celltypes()` |
| — | `m3_shutdown()` *(R only — releases the engine session)* |

## See also

- **[R API](api-r.md)** — the same surface for R (`library(m3)`).
- **Tutorials** — [Representation learning](notebooks/py_01_representation_learning.ipynb),
  [Patient prediction](notebooks/py_02_patient_prediction.ipynb),
  [Feature attribution](notebooks/py_03_attribution.ipynb),
  [Data augmentation](notebooks/py_04_augmentation.ipynb).
