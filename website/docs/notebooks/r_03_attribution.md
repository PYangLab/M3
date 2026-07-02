# Feature attribution

m3 explains its donor-level disease prediction with **end-to-end integrated gradients**, attributing the prediction back to **genes/proteins**, **cell types**, and **donors**. We train on the full demo reference, then call `m3_attribute(reference_labels = "HC")` (HC = the healthy integrated-gradients baseline) and visualise the rankings.

``` r
library(m3)
set.seed(0)
data <- m3_demo()
data
#> m3_dataset(n_cells=30534, batches=[B1, B2, B3], modalities=[rna:1000, adt:192])
```

## 2. Train (integration VAE + donor predictor on the full reference)

Attribution runs through the trained `(generator, corrector)`, so we provide `donor_key` + `celltype_key` (no held-out batch — we attribute on the full set). The donor-predictor knobs default to the from-scratch driver values.

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
  max_epochs       = 500L,
  seed = 0L
)
m3_capabilities(model)
#>      embedding    reconstruct predict_donors 
#>           TRUE           TRUE           TRUE
```

## 3. Attribute: Severe vs HC baseline

``` r
attr <- m3_attribute(model, reference_labels = "HC")   # n_steps defaults to 50
attr
#> m3_attribution(target=Severe, genes=1192, celltypes=17, donors=63)
cat("\ntop cell types (>= 200 cells per condition):\n")
#> 
#> top cell types (>= 200 cells per condition):
print(m3_top_celltypes(attr, min_cells_per_condition = 200L))
#>             celltype importance
#> 1     Mono_Classical   7.129031
#> 2  Mono_Nonclassical   6.961583
#> 3            CD8_Mem   6.789464
#> 4                 NK   6.645633
#> 5          CD8_Va7.2   6.418572
#> 6            CD4_Mem   6.375149
#> 7              B_Mem   6.215689
#> 8              T_Vd2   6.005471
#> 9          CD4_Naive   3.659189
#> 10         CD8_Naive   3.085082
#> 11           B_Naive   1.832521
```

### Per-celltype-balanced gene ranking (the publication recipe)

`attr$genes` is the **raw** ranking — `mean(|IG|)` over all cells. The publication recipe drops cell types with \< 200 cells in either condition, scores each gene by `mean(|gene x celltype IG|)` over the kept cell types, excludes housekeeping / ribosomal genes, and (optionally) restricts to one modality.

``` r
top100_rna <- m3_top_genes(attr, n = 100L, min_cells_per_condition = 200L, modality = "rna")
cat(sprintf("top-100 RNA genes (computed over %d balanced cell types):\n",
            top100_rna$n_celltypes_used[1]))
#> top-100 RNA genes (computed over 11 balanced cell types):
print(utils::head(top100_rna, 15))
#>    feature modality       score n_celltypes_used
#> 1     OAZ1      rna 0.002577195               11
#> 2    HSPA8      rna 0.002045409               11
#> 3  PIK3IP1      rna 0.002016037               11
#> 4     CD69      rna 0.001964119               11
#> 5     IL32      rna 0.001842893               11
#> 6     IL7R      rna 0.001761971               11
#> 7  TSC22D3      rna 0.001632039               11
#> 8    CXCR4      rna 0.001501254               11
#> 9     UCP2      rna 0.001479724               11
#> 10    JUNB      rna 0.001447082               11
#> 11     LTB      rna 0.001406194               11
#> 12  IFITM1      rna 0.001405208               11
#> 13   DDIT4      rna 0.001379280               11
#> 14   ANXA6      rna 0.001316347               11
#> 15  FAM65B      rna 0.001314393               11
```

## 4. Visualise — top genes, top cell types, gene x celltype heatmap

``` r
g <- utils::head(top100_rna, 20)
p1 <- ggplot(g, aes(stats::reorder(feature, score), score)) +
  geom_col(fill = "#4c72b0") + coord_flip() +
  labs(title = "Top-20 RNA genes (per-celltype-balanced, housekeeping excluded)",
       x = NULL, y = "mean |IG| across balanced cell types") +
  theme_classic() + theme(axis.text.y = element_text(size = 7))

c20 <- m3_top_celltypes(attr, min_cells_per_condition = 200L)
p2 <- ggplot(c20, aes(stats::reorder(celltype, importance), importance)) +
  geom_col(fill = "#dd8452") + coord_flip() +
  labs(title = "Cell-type importance (>= 200 cells per condition)", x = NULL, y = "importance") +
  theme_classic() + theme(axis.text.y = element_text(size = 8))
print(p1); print(p2)
```

<img src="../r_03_attribution_media/5d3b76c8399a535f6d2e37252fc3a6404f24aed4.png" width="1344" /><img src="../r_03_attribution_media/15db5918c1b56efbcebae5ba1519f4e3488b76a9.png" width="1344" />

### Gene x cell-type attribution heatmap (top-30 RNA genes)

``` r
gcm <- m3_gene_celltype_matrix(attr)                 # celltype x feature
top_features <- utils::head(top100_rna$feature, 30)
idx <- match(top_features, attr$feature_names)
sub <- gcm[, idx, drop = FALSE]
hm <- expand.grid(celltype = attr$celltype_names, feature = top_features,
                  stringsAsFactors = FALSE)
hm$value <- as.vector(sub)
hm$feature <- factor(hm$feature, levels = top_features)
hm$celltype <- factor(hm$celltype, levels = attr$celltype_names)
lim <- max(abs(hm$value), na.rm = TRUE)
ggplot(hm, aes(feature, celltype, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#053061", mid = "white", high = "#67001f",
                       midpoint = 0, limits = c(-lim, lim), name = "signed IG") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 6),
        axis.text.y = element_text(size = 7)) +
  labs(title = "Gene x cell-type attribution (top-30 RNA genes, balanced ranking)",
       x = NULL, y = NULL)
```

<img src="../r_03_attribution_media/23b9f87a749b7170693100e5571bef28638f4f91.png" width="1152" />

**Done.** End-to-end integrated-gradients attribution: ranked gene/protein, cell-type and donor importance for the Severe-vs-HC prediction.
