# Use your own data

The other tutorials start from the built-in demo. This one starts from
your own objects: count matrices that are stored separately per
modality, a `SingleCellExperiment`, or a Seurat object. It also covers
batches that do not all measure the same modalities.

``` r
library(m3)
set.seed(0)
```

## 0. Example data

So that there is something to work with, this chunk makes a small
synthetic RNA matrix, ADT matrix and metadata table. **Skip it when you
have your own.**

``` r
library(SingleCellExperiment)
sce  <- m3_example_sce(n_cells = 360)
rna  <- counts(sce)                         # genes x cells
adt  <- assay(altExp(sce, "ADT"))           # proteins x cells
meta <- as.data.frame(colData(sce))         # one row per cell
colnames(rna) <- colnames(adt) <- rownames(meta) <- sprintf("cell%03d", seq_len(ncol(rna)))
adt  <- adt[, sample(ncol(adt))]            # other cell order
dim(rna); dim(adt)
#> [1]  60 360
#> [1]  12 360
head(meta, 3)
#>         mergedcelltype cond_group Age_interval  sample_id batch
#> cell001             NK         HC        young s1_donor01    B1
#> cell002        CD8_Mem     Severe          old s1_donor04    B2
#> cell003      CD14_Mono         HC          old s1_donor07    B2
```

## 1. What M3 needs

|  |  |
|----|----|
| **Counts** | Raw, un-normalised counts for every modality, features in rows and cells in columns. M3 normalises internally. |
| **Modalities** | RNA, ADT and/or ATAC. |
| **Cells** | Within a batch, the same cells in every modality, in the same order. |
| **Features** | Unique names per modality, named the same way in every batch. |
| **Metadata columns** | A batch column (for example the sequencing run) and one or two condition columns (for example disease status). The donor-level tasks also need a cell-type column and a donor column. |

## 2. Separate matrices per modality

Keep the cells the modalities share, put them in one order, and pass a
list with `rna`, optionally `adt` and `atac`, and the metadata as `obs`.

``` r
cells <- intersect(colnames(rna), colnames(adt))
data <- m3_dataset(list(rna = rna[, cells], adt = adt[, cells], obs = meta[cells, ]))
data
#> m3_dataset(n_cells=360, batches=[B1, B2], modalities=[rna:60, adt:12])
```

The batches printed here come from the metadata column called `batch`.
If your batch column has another name, leave it as it is and pass it as
`batch_key = "run"` to `m3_train()`: the printed dataset then shows
`batches=not set`, and the printed model shows which column is the
batch. Without a batch column `m3_train()` stops with an error that
lists the metadata columns. Check that the printed dataset lists your
modalities.

## 3. Train and read out

`condition_keys` (one or two columns) tells the model which metadata
columns hold the conditions; it is required. With the condition and
batch columns alone, `m3_train()` fits the cell-level model, which is
all the embedding needs. `celltype_key` and `donor_key` are for the
donor-level tasks.

``` r
model <- m3_train(data, condition_keys = "cond_group")
#> Batch counts: {np.int64(0): np.int64(192), np.int64(1): np.int64(168)}
#> Minimum batch size: 168
#> Epoch 1, Validation Loss: 7.8255
#> Epoch 2, Validation Loss: 7.8842
#> Epoch 3, Validation Loss: 7.8457
#> ...  (294 epochs omitted)  ...
#> Epoch 298, Validation Loss: 7.4190
#> Epoch 299, Validation Loss: 7.4752
#> Epoch 300, Validation Loss: 7.5334
```

The embedding comes back in the model’s own row order. The cell names
are in the `obs_names` column of `m3_cell_metadata()`; use them to put
the rows in your order.

``` r
emb <- m3_embedding(model, part = "bio")
rownames(emb) <- m3_cell_metadata(model)$obs_names
emb <- emb[cells, ]
dim(emb)
#> [1] 360  28
```

`part = "bio"` includes the condition factors, so cells also separate by
condition; `part = "intrinsic"` leaves them out.

The model lives in the running R session and is not stored by
`saveRDS()` or `save.image()`. To keep it, write it with
[`m3_save(model, "model.m3")`](../api-r.md#m3_save) and read it back in
a later session with [`m3_load("model.m3", data)`](../api-r.md#m3_load),
which needs the same `data`. Read-outs are ordinary R objects:
`saveRDS(emb, "m3_embedding.rds")`.

## 4. SingleCellExperiment or Seurat objects

`m3_dataset()` also takes the object itself. In a
`SingleCellExperiment`, RNA is the `counts` assay and ADT / ATAC are
`altExp`s named `"ADT"` / `"ATAC"`.

``` r
m3_dataset(sce)
#> m3_dataset(n_cells=360, batches=[B1, B2], modalities=[rna:60, adt:12])
```

### Seurat

Start the m3 engine before Seurat is loaded; `library(m3)` alone does
not start it. If Seurat is loaded first, including by passing a Seurat
object to m3 as the very first m3 call, R can crash with
`*** caught segfault ***`; restart R in that case. A script that works
with Seurat objects therefore begins like this:

``` r
library(m3)
invisible(m3_dataset(m3_example_sce()))     # starts the m3 engine
library(Seurat)
```

RNA is read from the raw counts of the default assay, ADT / ATAC from
assays named `"ADT"` / `"ATAC"`. If your assay has another name, pass it
as `adt_exp =` / `atac_exp =` to `m3_dataset()`; an assay that is not
found is left out without an error, so check the modalities in the
printed dataset. A warning that the `scale.data` layer is empty can be
ignored.

``` r
so <- CreateSeuratObject(counts = as(rna, "dgCMatrix"), meta.data = meta)
so[["Protein"]] <- CreateAssayObject(counts = as(adt[, colnames(rna)], "dgCMatrix"))

data <- m3_dataset(so, adt_exp = "Protein")
data
#> m3_dataset(n_cells=360, batches=[B1, B2], modalities=[rna:60, adt:12])
model <- m3_train(data, condition_keys = "cond_group")
#> Batch counts: {np.int64(0): np.int64(192), np.int64(1): np.int64(168)}
#> Minimum batch size: 168
#> Epoch 1, Validation Loss: 7.8255
#> Epoch 2, Validation Loss: 7.8842
#> Epoch 3, Validation Loss: 7.8457
#> ...  (294 epochs omitted)  ...
#> Epoch 298, Validation Loss: 7.4190
#> Epoch 299, Validation Loss: 7.4752
#> Epoch 300, Validation Loss: 7.5334
```

Store the embedding as a reduction, matched by cell name, and continue
in Seurat.

``` r
emb <- m3_embedding(model, part = "bio")
rownames(emb) <- m3_cell_metadata(model)$obs_names
colnames(emb) <- paste0("M3_", seq_len(ncol(emb)))

so[["m3"]] <- CreateDimReducObject(emb[colnames(so), ], key = "M3_", assay = "RNA")
so <- RunUMAP(so, reduction = "m3", dims = 1:ncol(emb), verbose = FALSE)
so
#> An object of class Seurat 
#> 72 features across 360 samples within 2 assays 
#> Active assay: RNA (60 features, 0 variable features)
#>  1 layer present: counts
#>  1 other assay present: Protein
#>  2 dimensional reductions calculated: m3, umap
```

## 5. Batches with different modalities

Build one dataset per batch, each with the modalities it has, and stack
them with `m3_concat()`. Here batch `B2` has RNA only. `batch =` names
each batch, because the metadata passed here has no `batch` column.

``` r
b1 <- rownames(meta)[meta$batch == "B1"]
b2 <- rownames(meta)[meta$batch == "B2"]
obs_cols <- setdiff(names(meta), "batch")

d1 <- m3_dataset(list(rna = rna[, b1], adt = adt[, b1], obs = meta[b1, obs_cols]), batch = "B1")
d2 <- m3_dataset(list(rna = rna[, b2],                  obs = meta[b2, obs_cols]), batch = "B2")
mosaic <- m3_concat(list(d1, d2))
mosaic
#> m3_dataset(n_cells=360, batches=[B1, B2], modalities=[adt:12, rna:60])
```

The printed dataset lists every modality that any batch has. To see
which cells have ADT, look at the row sums of the ADT matrix, which are
zero where it was not measured:

``` r
has_adt <- rowSums(m3_dataset_matrix(mosaic, "adt")) > 0
table(m3_dataset_obs(mosaic)$batch, has_adt)
#>     has_adt
#>      FALSE TRUE
#>   B1     0  192
#>   B2   168    0
```

Training is the same call, and the embedding covers the cells of both
batches.

``` r
model <- m3_train(mosaic, condition_keys = "cond_group")
#> Batch counts: {np.int64(0): np.int64(192), np.int64(1): np.int64(168)}
#> Minimum batch size: 168
#> Epoch 1, Validation Loss: 7.1456
#> Epoch 2, Validation Loss: 7.2440
#> Epoch 3, Validation Loss: 7.1541
#> ...  (294 epochs omitted)  ...
#> Epoch 298, Validation Loss: 6.7170
#> Epoch 299, Validation Loss: 6.7963
#> Epoch 300, Validation Loss: 6.8622
emb <- m3_embedding(model, part = "bio")
table(m3_cell_metadata(model)$batch)
#> 
#>  B1  B2 
#> 192 168
dim(emb)
#> [1] 360  28
```

With Seurat objects it is one object per batch, without the protein
assay where it was not measured:

``` r
s1 <- subset(so, cells = b1)
s2 <- subset(so, cells = b2)
s2[["Protein"]] <- NULL                       # B2 has RNA only
s1$batch <- NULL; s2$batch <- NULL

m3_concat(list(m3_dataset(s1, batch = "B1", adt_exp = "Protein"),
               m3_dataset(s2, batch = "B2")))
#> m3_dataset(n_cells=360, batches=[B1, B2], modalities=[adt:12, rna:60])
```

A list needs an `rna` element, so a batch without RNA cannot be built
from a list.
