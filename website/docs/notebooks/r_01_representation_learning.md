# Cell-level representation learning

This tutorial trains an **m3** integration model on the multi-batch, multimodal (RNA + ADT) **Liu et al.** COVID-19 demo and produces a batch-corrected, condition-aware **cell embedding**.

m3’s latent is **disentangled**: a cell-intrinsic part, a small per-condition part, and a batch part. `m3_embedding(part = "bio")` returns the biology (intrinsic + conditions, batch removed) — the representation you use for clustering / UMAP.

``` r
library(m3)
set.seed(0)
```

## 1. Load the demo dataset

`m3_demo()` returns the same stratified subsample of the three Liu batches the Python package ships, as an `m3_dataset` (RNA 1000 HVG + ADT, batches `B1/B2/B3`).

``` r
data <- m3_demo()
data
#> m3_dataset(n_cells=30534, batches=[B1, B2, B3], modalities=[rna:1000, adt:192])
```

## 2. Build and train the integration model

We declare the column roles. For *pure* representation learning we set `donor_prediction = FALSE` so only the integration VAE is trained (the donor-level disease predictor is Tutorial 2). `embedding_dim` is the latent width. `batch_key` (default `"batch"`) names the obs column of batch labels (B1/B2/B3) the VAE balances and corrects across — a real input even here. `seed = 0` makes the (otherwise unseeded) Stage-1 VAE reproducible.

``` r
#To save time, users can set max_epochs to 100 for test.
model <- m3_train(
  data,
  condition_keys = c("cond_group", "Age_interval"),
  celltype_key   = "mergedcelltype",
  batch_key      = "batch",
  embedding_dim  = 30L,
  donor_prediction = FALSE,
  max_epochs = 500L,
  seed = 0L
)
#> Using 'matrix/data'
#> torch.Size([1000, 7163])
#> Using 'matrix/data'
#> torch.Size([1000, 10789])
#> Using 'matrix/data'
#> torch.Size([1000, 12582])
#> Using 'matrix/data'
#> torch.Size([192, 7163])
#> Using 'matrix/data'
#> torch.Size([192, 10789])
#> Using 'matrix/data'
#> torch.Size([192, 12582])
#> Batch counts: {0: 7163, 1: 10789, 2: 12582}
#> Minimum batch size: 7163
#> Epoch 1, Validation Loss: 5.1807
#> Epoch 2, Validation Loss: 5.1167
#> Epoch 3, Validation Loss: 5.0486
#> ...  (494 epochs omitted)  ...
#> Epoch 498, Validation Loss: 0.5032
#> Epoch 499, Validation Loss: 0.5039
#> Epoch 500, Validation Loss: 0.5011
#> Using 'matrix/data'
#> torch.Size([1000, 7163])
#> Using 'matrix/data'
#> torch.Size([1000, 10789])
#> Using 'matrix/data'
#> torch.Size([1000, 12582])
#> Using 'matrix/data'
#> torch.Size([192, 7163])
#> Using 'matrix/data'
#> torch.Size([192, 10789])
#> Using 'matrix/data'
#> torch.Size([192, 12582])
model
#> m3_model
#>   modalities     : rna, adt (embedding_dim=30)
#>   condition keys : cond_group, Age_interval  (target: cond_group)
#>   roles          : celltype=mergedcelltype  donor=-  batch=batch
#>   capabilities   : embedding, reconstruct
m3_capabilities(model)
#>      embedding    reconstruct predict_donors 
#>           TRUE           TRUE          FALSE
```

## 3. Extract the disentangled embeddings

`part = "bio"` = cell-intrinsic + condition latents (batch removed) — the integrated biological representation. `part = "batch"` isolates the batch latent.

``` r
emb_bio       <- m3_embedding(model, part = "bio")
emb_intrinsic <- m3_embedding(model, part = "intrinsic")
emb_batch     <- m3_embedding(model, part = "batch")
meta <- m3_cell_metadata(model)
cat(sprintf("bio: %s | intrinsic: %s | batch: %s\n",
            paste(dim(emb_bio), collapse = "x"),
            paste(dim(emb_intrinsic), collapse = "x"),
            paste(dim(emb_batch), collapse = "x")))
#> bio: 30534x28 | intrinsic: 30534x24 | batch: 30534x2
```

## 4. Save the embedding + metadata

m3 returns plain matrices, so persist them however you like — here as CSV and an `.rds`.

``` r
out <- file.path(tempdir(), "m3_tut1")
dir.create(out, showWarnings = FALSE)
utils::write.csv(emb_bio, file.path(out, "embedding_bio.csv"), row.names = FALSE)
utils::write.csv(meta, file.path(out, "cell_metadata.csv"), row.names = FALSE)
saveRDS(list(embedding = emb_bio, metadata = meta), file.path(out, "embedding_bio.rds"))
cat("saved embedding to", out, "\n")
#> saved embedding to /tmp/RtmpavmnOE/m3_tut1
```

## 5. Visualise with UMAP

We project the *bio* embedding with UMAP. Good integration = cell types form clean clusters while batches are mixed.

``` r
xy <- m3_umap(emb_bio, method = "scanpy", n_neighbors = 15L, device = model$device)
df <- data.frame(UMAP1 = xy[, 1], UMAP2 = xy[, 2],
                 celltype = factor(meta$mergedcelltype),
                 batch    = factor(meta$batch),
                 condition = factor(meta$cond_group))

plt <- function(colour, title, legend = TRUE) {
  ggplot(df, aes(UMAP1, UMAP2, colour = .data[[colour]])) +
    geom_point(size = 0.4, alpha = 0.8) +
    theme_classic() +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          legend.position = if (legend) "right" else "none") +
    guides(colour = guide_legend(override.aes = list(size = 2))) +
    labs(title = title, colour = NULL)
}
print(plt("celltype",  "Cell type (biology preserved)"))
```

<img src="../r_01_representation_learning_media/dba482b006a975e8e0a61155ffd05c8c04f35ff9.png" width="1536" />

``` r
print(plt("batch",     "Batch (batch mixed)"))
```

<img src="../r_01_representation_learning_media/172904583ccd9f0b8084d9782089b26d6bcf352d.png" width="1536" />

``` r
print(plt("condition", "Condition"))
```

<img src="../r_01_representation_learning_media/f958d3366ee290d0aad44b119a0823511cc7a2ed.png" width="1536" />

### Batch latent (the part m3 *removed* from “bio”)

Colouring a UMAP of the **batch** latent by batch shows the batch structure m3 isolates into the batch dimension and keeps out of the biological embedding.

``` r
xy_b <- m3_umap(emb_batch, method = "scanpy", n_neighbors = 15L, device = model$device)
ggplot(data.frame(UMAP1 = xy_b[, 1], UMAP2 = xy_b[, 2], batch = factor(meta$batch)),
       aes(UMAP1, UMAP2, colour = batch)) +
  geom_point(size = 0.4, alpha = 0.8) + theme_classic() +
  theme(axis.text = element_blank(), axis.ticks = element_blank()) +
  labs(title = "Batch latent — batch signal", colour = NULL)
```

<img src="../r_01_representation_learning_media/0ed26f77a17ca6ddfeed86abdda7877dbc9d8664.png" width="576" />

**Done.** We trained one m3 model on three batches and obtained an integrated, condition-aware cell embedding plus UMAPs. Tutorial 2 builds on the same model object to predict donor-level disease status.
