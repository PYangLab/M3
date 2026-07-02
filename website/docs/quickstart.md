---
tags:
  - getting-started
---

# Quickstart

This page runs M3 end to end on the **built-in demo dataset** — no downloads.
Every snippet uses the real public API; for fully rendered, executed versions
(with figures and outputs) see the [tutorials](notebooks/py_01_representation_learning.ipynb).

!!! tip "Prerequisites"
    Install PyTorch and M3 (from GitHub) — see [Installation](installation.md) —
    plus the optional plotting extras `pip install scanpy umap-learn matplotlib`.

## 0. Load the demo data

`m3.datasets.liu_demo()` returns a ready-to-use [`Dataset`](api-python.md#the-dataset-container):
a stratified subsample of the Liu *et al.* COVID-19 dataset
(RNA + ADT, three batches).

```python
import m3

data = m3.datasets.liu_demo()
print(data)
# Dataset(n_cells=30534, batches=['B1', 'B2', 'B3'], modalities=[rna:1000, adt:192])
```

The relevant `obs` columns are `cond_group` (`HC` / `Severe`),
`mergedcelltype`, `sample_id` (donor), `batch`, and `Age_interval`.

## 1. Representation learning

Build a model by declaring which `obs` columns play which role, then train. For
pure embedding, switch the donor predictor off.

```python
model = m3.M3(
    data,
    condition_keys=["cond_group", "Age_interval"],
    celltype_key="mergedcelltype",
    embedding_dim=30,
)
model.train(max_epochs=500, donor_prediction=False)

emb = model.embedding(part="bio")        # (n_cells, d) condition-aware embedding
meta = model.cell_metadata               # row-aligned metadata
print(emb.shape)
```

Feed `emb` straight into a UMAP coloured by cell type (biology preserved) and by
batch (mixed). See
[Tutorial 1](notebooks/py_01_representation_learning.ipynb) for the plotting code.

## 2. Patient-level disease prediction

Hold a batch out as an unlabelled **query**, supply `donor_key` + `celltype_key`
(which auto-enables the donor predictor), and tune it via `donor_predictor`.

```python
model = m3.M3(
    data,
    condition_keys=["cond_group", "Age_interval"],
    target_condition="cond_group",
    celltype_key="mergedcelltype",
    donor_key="sample_id",
    batch_key="batch",
    held_out=["B3"],                 # B3 labels are masked during training
    embedding_dim=30,
)
model.train(
    max_epochs=500,
    donor_predictor={"glr": 3e-3, "n_epochs": 120, "adv_max": 10,
                     "adv_warmup": 7, "n_disc": 21, "patient_w": 10},
)

preds = model.predict_donors()           # held-out donors only
print(preds[["donor", "predicted_label", "prob_Severe"]])
```

`predict_donors()` returns one row per held-out donor with the predicted label
and a probability per class. The corrected per-donor vectors are available from
[`donor_embedding()`](api-python.md#m3donor_embedding) for a patient-level UMAP. Full
walkthrough: [Tutorial 2](notebooks/py_02_patient_prediction.ipynb).

## 3. Feature attribution

Train with the donor predictor (the default when `donor_key` + `celltype_key` are
set), then attribute the disease condition against a healthy baseline.

```python
model = m3.M3(
    data,
    condition_keys=["cond_group", "Age_interval"],
    target_condition="cond_group",
    celltype_key="mergedcelltype",
    donor_key="sample_id",
    batch_key="batch",
    embedding_dim=30,
)
model.train(max_epochs=500)

attr = model.attribute(reference_labels=["HC"])      # Severe vs HC
attr.top_celltypes(min_cells_per_condition=200)      # ranked cell types
attr.top_genes(n=100, min_cells_per_condition=200, modality="rna")
```

`top_genes` applies the publication recipe (per–cell-type balancing,
housekeeping-gene exclusion); `attr.gene_celltype_matrix` is the signed
cell-type × gene matrix behind it. See
[Tutorial 3](notebooks/py_03_attribution.ipynb).

## 4. Data augmentation

Synthesise new donors per condition — optionally stratified by batch — or
posterior-resample cells 1:1.

```python
aug = model.augment(conditions=["HC", "Severe"], n_donors=[3, 3], tau=0.8)
syn_rna = aug["expression"]["rna"]       # (n_synth_cells, n_genes)
syn_obs = aug["obs"]                     # condition / donor / cell-type labels

gen = model.generate(tau=0.8)            # one synthetic cell per reference cell
```

Pass `batch="B1"` to `augment` to template only from that batch. Full
example with the batch-stratified UMAP:
[Tutorial 4](notebooks/py_04_augmentation.ipynb).

---

## Where to next

- The four [tutorials](notebooks/py_01_representation_learning.ipynb) — the same
  workflow, fully executed with figures.
- The [API reference](api-python.md) — every argument and return type.
- [How M3 works](overview.md) — the model and the six tasks.
