# Donor-level disease prediction

m3 can predict a **donor's** disease status from their cells, in a leave-one-batch-out setting: we hold out one batch (`B3`), hide its donors' disease labels during training, and predict them at the end. `m3_train(...)` fits the integration model and the donor predictor in one call; `m3_predict_donors()` then returns per-donor class probabilities.

## 1. Load the demo dataset

``` r
library(m3)
set.seed(0)
data <- m3_demo()
data
#> m3_dataset(n_cells=30534, batches=[B1, B2, B3], modalities=[rna:1000, adt:192])
```

## 2. Build + train the model with a held-out batch

On top of the columns from the representation-learning tutorial, donor prediction needs a few more:

- `target_condition`, the condition to predict (here `cond_group`, i.e. disease).
- `donor_key`, the column identifying each donor, so cells are grouped per donor.
- `held_out`, the batch(es) to hold back; these donors' `target_condition` labels are hidden during training and predicted at the end.

The donor predictor summarises each donor as a cell-type-balanced mean of the latent embedding, then removes batch effects from it by adversarial domain adaptation: a **corrector** reshapes the profile while a **batch discriminator** tries to identify the batch from it, and the corrector is trained to predict disease while preventing the discriminator from succeeding. The `donor_predictor` settings:

- `patient_w` — weight of the donor objective relative to the cell-level VAE loss.
- `adv_max` — the maximum strength of batch removal.
- `adv_warmup` — number of epochs over which batch-removal strength ramps from 0 to `adv_max`.
- `n_disc` — discriminator updates per corrector update.
- `glr`, `n_epochs` — the corrector's learning rate and number of epochs.

`m3_reference_vocab(model)` shows the label set the model learned to predict (the held-out donors' labels never enter it).

``` r
#To save time, users can set max_epochs to 100 for test.
model <- m3_train(
  data,
  condition_keys   = c("cond_group", "Age_interval"),
  target_condition = "cond_group",
  celltype_key     = "mergedcelltype",
  batch_key        = "batch",
  donor_key        = "sample_id",
  held_out         = "B3",
  embedding_dim    = 30L,
  max_epochs       = 500L,
  donor_predictor  = list(glr = 3e-3, n_epochs = 120L, adv_max = 10L,
                          adv_warmup = 7L, n_disc = 21L, patient_w = 10L),
  seed = 0L
)
cat("reference vocab (query labels never enter it):",
    paste(m3_reference_vocab(model)$cond_group, collapse = ", "), "\n")
#> reference vocab (query labels never enter it): HC, Severe
```

## 3. Predict the held-out donors

`m3_predict_donors(model)` returns one row per held-out donor: the predicted label and the probability of each class.

``` r
preds <- m3_predict_donors(model)
cat("query donors:", nrow(preds), "\n")
#> query donors: 23
print(preds)
#>           sample_id is_reference predicted_label      prob_HC  prob_Severe
#> 1  B3_HGR0000051_T0        FALSE          Severe 6.211422e-07 0.9999994040
#> 2  B3_HGR0000051_T1        FALSE          Severe 3.821796e-05 0.9999617338
#> 3  B3_HGR0000051_T2        FALSE          Severe 1.434713e-04 0.9998564720
#> 4  B3_HGR0000051_T3        FALSE          Severe 8.502320e-05 0.9999150038
#> 5  B3_HGR0000101_T0        FALSE          Severe 1.689571e-04 0.9998309612
#> 6  B3_HGR0000101_T1        FALSE          Severe 1.553886e-06 0.9999984503
#> 7  B3_HGR0000101_T3        FALSE          Severe 4.099387e-05 0.9999589920
#> 8  B3_HGR0000102_T0        FALSE          Severe 5.886243e-06 0.9999941587
#> 9  B3_HGR0000102_T1        FALSE          Severe 1.414257e-06 0.9999985695
#> 10 B3_HGR0000134_T0        FALSE          Severe 2.599255e-05 0.9999740124
#> 11 B3_HGR0000134_T1        FALSE          Severe 3.862466e-05 0.9999613762
#> 12 B3_HGR0000135_T0        FALSE          Severe 3.230921e-06 0.9999967813
#> 13 B3_HGR0000135_T1        FALSE          Severe 2.058625e-07 0.9999997616
#> 14 B3_HGR0000142_T0        FALSE          Severe 2.676351e-05 0.9999731779
#> 15 B3_HGR0000392_T0        FALSE          Severe 1.773454e-06 0.9999982119
#> 16 B3_HGR0000392_T3        FALSE          Severe 1.556560e-05 0.9999843836
#> 17 B3_HGR0000429_T1        FALSE          Severe 1.968242e-05 0.9999803305
#> 18 B3_HGR0000429_T2        FALSE          Severe 7.869442e-05 0.9999213219
#> 19 B3_HGR0000430_T1        FALSE          Severe 3.379085e-04 0.9996620417
#> 20       B3_SHD1_HC        FALSE              HC 9.306676e-01 0.0693323463
#> 21       B3_SHD3_HC        FALSE              HC 9.538549e-01 0.0461451486
#> 22       B3_SHD5_HC        FALSE              HC 9.995824e-01 0.0004175995
#> 23       B3_SHD6_HC        FALSE              HC 9.905715e-01 0.0094285561
```

## 4. Evaluate against the held-out truth

The held-out donors' real `cond_group` lives in the metadata (never shown to the model). We join it back to score accuracy.

``` r
obs <- m3_dataset_obs(data)
truth <- unique(obs[, c("sample_id", "cond_group")])
truth <- stats::setNames(as.character(truth$cond_group), as.character(truth$sample_id))
preds$true_label <- truth[as.character(preds$sample_id)]
acc <- mean(preds$predicted_label == preds$true_label)
cat(sprintf("held-out accuracy = %.3f\n", acc))
#> held-out accuracy = 1.000
```

## 5. Patient-level embedding

`m3_donor_embedding()` returns one vector per donor, the donor-level representation the model actually classifies. We UMAP it and colour by reference/query, true label, and whether the held-out prediction was correct.

``` r
demb <- m3_donor_embedding(model)
info <- m3_predict_donors(model, include_reference = TRUE)
info <- info[match(demb$sample_id, info$sample_id), ]
info$true_label <- truth[as.character(demb$sample_id)]
info$set     <- ifelse(info$is_reference, "reference", "query")
info$correct <- ifelse(info$is_reference, "reference",
                       ifelse(info$predicted_label == info$true_label, "correct", "wrong"))

X  <- as.matrix(demb[, grep("^m3_", colnames(demb))])
xy <- m3_umap(X, method = "umap", n_neighbors = 15L, random_state = 0L,
              device = model$device)
pdf <- data.frame(UMAP1 = xy[, 1], UMAP2 = xy[, 2],
                  set = factor(info$set), true_label = factor(info$true_label),
                  correct = factor(info$correct))

pl <- function(colour, title) {
  ggplot(pdf, aes(UMAP1, UMAP2, colour = .data[[colour]])) +
    geom_point(size = 3, alpha = 0.85) + theme_classic() +
    theme(axis.text = element_blank(), axis.ticks = element_blank()) +
    labs(title = title, colour = NULL)
}
print(pl("set",        "Patient embedding — set"))
```

<img src="../r_02_patient_prediction_media/226d25d20953d65d75a80fc3402e223933193b70.png" width="1536" />

``` r
print(pl("true_label", "Patient embedding — true_label"))
```

<img src="../r_02_patient_prediction_media/b2c1395cc2fc45a3baf8549bb3334f09fedfa9f0.png" width="1536" />

``` r
print(pl("correct",    "Patient embedding — correct"))
```

<img src="../r_02_patient_prediction_media/6e246b53c0803f09cd89b09bb1173c4ff2d79999.png" width="1536" />

**Done.** Leave-one-batch-out donor prediction, evaluated against the held-out truth, with a patient-level embedding.
