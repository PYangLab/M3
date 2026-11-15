---
tags:
  - faq
---

# Frequently asked questions

## Your own data

### What input does M3 take?

Raw counts for each modality (`rna`, `adt`, `atac`) and a per-cell table with a
batch column, one or two condition columns and a cell-type column. The
"Use your own data" tutorial goes from files or objects to a trained model, in
[Python](notebooks/py_05_own_data.ipynb) and in [R](notebooks/r_05_own_data.md).

### My RNA and ADT (or ATAC) are in separate files

Read both, keep the shared cells in one order, and pass one matrix per modality
to [`m3.read_matrix`](api-python.md#m3read_matrix) (Python) or
[`m3_dataset()`](api-r.md#m3_dataset) (R). See
[Use your own data](notebooks/py_05_own_data.ipynb). MuData works the same way:
pass `mdata["rna"].X` and the other modalities to `read_matrix`.

### My batches do not all measure the same modalities

Build each batch with the modalities it has and stack them with
[`m3.concat`](api-python.md#m3concat) / [`m3_concat()`](api-r.md#m3_concat). See
[Mosaic integration](notebooks/py_06_mosaic_integration.ipynb), or section 5 of
[Use your own data (R)](notebooks/r_05_own_data.md).

### My `.X` is normalised

M3 needs raw counts and warns when it finds non-integer values. If the counts
are in a layer, pass the layer instead of `.X`, as in
[Use your own data](notebooks/py_05_own_data.ipynb).

### I have no cell-type labels

A cell-type column is required for training. The labels are used by the donor
predictor, attribution and augmentation. The cell-level model
(`donor_prediction=False`) does not use them, so for an embedding you can add a
placeholder column to `obs`, for example `obs["celltype"] = "unknown"`, and pass
that.

### My study has one condition, or more than two

`condition_keys` is required and takes one or two columns. A column with a
single value is accepted.

### I have a modality other than RNA, ADT or ATAC

Not supported. The model uses the modalities named `rna`, `adt` and `atac`, and
one with any other name is left out without an error.

### I work in Seurat or SingleCellExperiment

Use the R package: [Use your own data (R)](notebooks/r_05_own_data.md). With
Seurat, start the m3 engine before Seurat is loaded, as shown there; in the
other order R can crash with a segmentation fault.

## Training and results

### How long does training take?

At the default 300 epochs on a laptop CPU, a few minutes for 3,000 cells and 15
to 20 minutes for the 30,534-cell demo. A CUDA GPU is used automatically when
present.

### How do I know training has run long enough?

Training prints the validation loss after every epoch, and `max_epochs` sets the
number of epochs (default 300). An epoch on a small dataset has fewer update
steps than one on a large dataset, so after the same number of epochs a small
dataset is further from convergence: on the 3,000-cell demo the loss is still
falling steeply at epoch 300.

### The embedding has 28 columns, not 30

`embedding_dim` (default 30) is the size of the whole latent. Its last two
dimensions are the batch factor, which the `"bio"` embedding leaves out.

### Cells separate by condition in the embedding

`part="bio"` is the intrinsic factor plus the condition factors, so it keeps the
differences between conditions. Use `part="intrinsic"` for an embedding without
them: that is the one to look at for how well cell types line up across batches.
To judge `"bio"`, use labels that combine cell type and condition.

### How do I make a run reproducible?

Training is seeded (`seed`, default `0`). The same data in the same cell order,
with the same settings and software versions, gives an identical embedding. A
different cell order or different software versions change the numbers.

### Can I save a trained model?

Yes. [`model.save("model.m3")`](api-python.md#m3save) writes the trained weights
and settings, and [`m3.M3.load("model.m3", data)`](api-python.md#m3load) reads
them back without training again. The file does not hold the data, so `load`
needs the same `Dataset` the model was trained on. In R:
[`m3_save()`](api-r.md#m3_save) and [`m3_load()`](api-r.md#m3_load).

### The embedding rows are not in my cell order

Read-outs are grouped by batch and follow `model.cell_metadata`, not the input.
Match them by the `obs_names` column, as in Use your own data
([Python](notebooks/py_05_own_data.ipynb), [R](notebooks/r_05_own_data.md)).

### Are the `reconstruct()` values counts?

No. They are on the scale the model trains on: `log2(count / size + 1)`, where
`size` is the cell's total count in that modality divided by the mean total over
all cells.
