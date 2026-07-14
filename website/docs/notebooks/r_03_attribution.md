# Feature attribution

Once m3 can predict disease, attribution asks which genes, cells and cell types drove that prediction. We train on the full reference, then trace each prediction back to the genes, cells, and cell types behind it.

## 1. Load the demo dataset

``` r
library(m3)
set.seed(0)
data <- m3_demo()
data
#> m3_dataset(n_cells=30534, batches=[B1, B2, B3], modalities=[rna:1000, adt:192])
```

## 2. Build and train the model

Same setup as the patient-prediction tutorial, minus the held-out batch (we attribute on the full reference). `m3_train(...)` fits both the integration model and the disease predictor; attribution explains that predictor.

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

## 3. Attribute and rank the cell types

`m3_attribute(model, reference_labels = "HC")` scores how much each gene, cell, and cell type pushed the prediction away from the reference label (HC, i.e. healthy). Here we look at the **top cell types**: `m3_top_celltypes(...)` ranks them, keeping only cell types with enough cells in each condition so a small group can't mislead the ranking.

``` r
attr <- m3_attribute(model, reference_labels = "HC")   # n_steps defaults to 50
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

## 4. Top genes

`m3_top_genes(...)` ranks the genes by how strongly they drove the prediction. `min_cells_per_condition` keeps only cell types with enough cells in each condition (so a tiny group can't dominate the ranking), and housekeeping / ribosomal genes are dropped.

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

**Done.** The genes, cells, and cell types behind m3's Severe-vs-HC prediction.
