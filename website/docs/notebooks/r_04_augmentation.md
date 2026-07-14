# Data augmentation

m3's decoder is generative: by sampling the trained model we can synthesise new **donors** (for augmenting small cohorts) and new **cells**. This tutorial makes synthetic donors per condition with `m3_augment()`, checks they match the real data, and resamples cells with `m3_generate()`.

## 1. Load the demo dataset

``` r
library(m3)
set.seed(0)
data <- m3_demo()
data
#> m3_dataset(n_cells=30534, batches=[B1, B2, B3], modalities=[rna:1000, adt:192])
```

## 2. Build and train the model

Augmentation only needs the generator, so we skip the donor predictor (`donor_prediction = FALSE`) to save time.

``` r
#To save time, users can set max_epochs to 100 for test.
model <- m3_train(
  data,
  condition_keys   = c("cond_group", "Age_interval"),
  target_condition = "cond_group",
  celltype_key     = "mergedcelltype",
  batch_key        = "batch",
  donor_key        = "sample_id",
  embedding_dim    = 30L,
  donor_prediction = FALSE,
  max_epochs       = 500L,
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
```

## 3. Synthesize new donors per condition

`m3_augment(...)` samples the trained model to create brand-new synthetic donors for the conditions you ask for. It returns a list with `expression` (per-modality synthetic cells) and `obs` (their condition and donor labels).

- `conditions` / `n_donors`: which conditions to synthesise, and how many new donors for each.
- `batch`: draw templates from one batch only, so the new donors carry that batch's character (used in the check below); omit it to draw from all batches.

``` r
aug <- m3_augment(model, conditions = c("HC", "Severe"), n_donors = c(3L, 3L), tau = 0.8)
syn_rna <- aug$expression$rna
cat("synthetic cells:", paste(dim(syn_rna), collapse = "x"), "\n")
#> synthetic cells: 3501x1000
print(table(aug$obs$cond_group))
#> 
#>     HC Severe 
#>   2609    892
```

## 4. Check against the real data

Do the synthetic donors look like real ones? We compare in the model's reconstruction space (where the synthetic cells live). A UMAP per batch and modality, one dot per sample: each sample is summarised by its mean expression within every shared cell type (so composition differences don't confound), then features are standardised. Real donors are open circles, m3-generated donors are crosses, colour marks the condition. Synthetic samples should sit with the real samples of the same condition in the same batch.

``` r
real_recon <- m3_reconstruct(model, remove_batch = FALSE)$rna
real_recon[is.na(real_recon)] <- 0
m_real <- colMeans(real_recon)
m_syn  <- colMeans(replace(syn_rna, is.na(syn_rna), 0))
pearson <- stats::cor(m_real, m_syn)
cat(sprintf("per-gene mean expression Pearson r = %.3f\n", pearson))
#> per-gene mean expression Pearson r = 0.998
```

``` r
# Compare samples in the model's reconstruction (decoder) space, the same space the
# generated cells live in. Each sample is summarised per shared cell type (so
# composition differences don't confound), then features are standardised.
recon <- m3_reconstruct(model, remove_batch = FALSE)
obs <- m3_cell_metadata(model)          # row-aligned with the reconstruction
real_by_mod <- list(RNA = replace(recon$rna, is.na(recon$rna), 0),
                    ADT = replace(recon$adt, is.na(recon$adt), 0))

# one row per sample: mean expression within each cell type shared by all samples
celltype_pseudobulk <- function(X, samples, celltypes) {
  keys <- unique(samples)
  shared <- Reduce(intersect, lapply(keys, function(k) unique(celltypes[samples == k])))
  M <- t(vapply(keys, function(k)
    unlist(lapply(shared, function(ct) colMeans(X[samples == k & celltypes == ct, , drop = FALSE]))),
    numeric(ncol(X) * length(shared))))
  list(M = M, keys = keys)
}

# standardise each feature, then a Euclidean UMAP of the samples
embed_2d <- function(pb, dev) {
  Z <- scale(pb)
  Z <- Z[, colSums(is.na(Z)) == 0, drop = FALSE]     # drop zero-variance features
  m3_umap(Z, method = "umap", n_neighbors = min(10L, nrow(Z) - 1L),
          min_dist = 0.3, metric = "euclidean", random_state = 0L, device = dev)
}

batches <- sort(unique(as.character(obs$batch)))
N_DONORS <- 5L
panels <- list()
for (ci in seq_along(batches)) {
  b <- batches[ci]
  in_b <- as.character(obs$batch) == b

  # augment templated on THIS batch, so the synthetic donors carry its character
  ag <- m3_augment(model, conditions = c("HC", "Severe"),
                   n_donors = c(N_DONORS, N_DONORS), batch = b, tau = 0.8, seed = 42L + ci - 1L)

  # real and synthetic cells share obs keys: sample_id (donor), cond_group, mergedcelltype
  samples <- c(as.character(obs$sample_id[in_b]), paste0(ag$obs$cond_group, "/", ag$obs$sample_id))
  celltypes <- c(as.character(obs$mergedcelltype[in_b]), as.character(ag$obs$mergedcelltype))
  cond <- c(as.character(obs$cond_group[in_b]), as.character(ag$obs$cond_group))
  is_real <- c(rep(TRUE, sum(in_b)), rep(FALSE, nrow(ag$obs)))
  cond_of <- tapply(cond, samples, function(x) x[1])
  real_of <- tapply(is_real, samples, function(x) x[1])

  for (md in c("RNA", "ADT")) {
    X <- rbind(real_by_mod[[md]][in_b, , drop = FALSE], ag$expression[[tolower(md)]])
    pb <- celltype_pseudobulk(X, samples, celltypes)
    xy <- embed_2d(pb$M, model$device)
    panels[[length(panels) + 1L]] <- data.frame(
      UMAP1 = xy[, 1], UMAP2 = xy[, 2],
      cond = cond_of[pb$keys], kind = ifelse(real_of[pb$keys], "real", "generated"),
      panel = paste0("Batch ", b, " (", md, ")"))
  }
}
plot_df <- do.call(rbind, panels)
cols <- c(HC = "#E64B35", Severe = "#4DBBD5")
ggplot() +
  geom_point(data = subset(plot_df, kind == "real"),
             aes(UMAP1, UMAP2, colour = cond), shape = 1, size = 3, stroke = 1.2) +
  geom_point(data = subset(plot_df, kind == "generated"),
             aes(UMAP1, UMAP2, colour = cond), shape = 4, size = 3, stroke = 1.4) +
  scale_colour_manual(values = cols, name = NULL) +
  facet_wrap(~ panel, scales = "free", ncol = length(batches)) +
  labs(title = "Liu et al — sample-level UMAP: real (o) vs m3 generated (x)") +
  theme_classic() +
  theme(axis.text = element_blank(), axis.ticks = element_blank())
```

<img src="../r_04_augmentation_media/2e1065f8e75a330fa47c1201754c8e81a0d7aae6.png" width="1728" />

## 5. Posterior-resampled cells (`generate`)

`m3_generate()` returns one synthetic cell per real cell (a 1:1 posterior resample), handy for noise-augmenting a training set at the cell level.

``` r
gen <- m3_generate(model, tau = 0.8)
cat("generated:", paste(names(gen), sapply(gen, function(m) paste(dim(m), collapse = "x")),
                        sep = "=", collapse = "  "), "\n")
#> generated: rna=30534x1000  adt=30534x192
```

**Done.** Synthetic donors per condition (checked against the real data) plus a 1:1 posterior resample of the cells.
