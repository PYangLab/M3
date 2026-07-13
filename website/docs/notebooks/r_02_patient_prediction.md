# Donor-level disease prediction

m3’s headline task: predict **donor/patient-level disease status** from multimodal single-cell data, in a **leak-safe leave-one-batch-out** setting. We hold out one batch (`B3`); its donors’ disease labels are masked during training and predicted at the end. One `m3_model` trains the integration VAE **and** the donor-level adversarial predictor; `m3_predict_donors()` returns per-donor class probabilities.

## 1. Load the demo dataset

``` r
library(m3)
set.seed(0)
data <- m3_demo()
data
#> m3_dataset(n_cells=30534, batches=[B1, B2, B3], modalities=[rna:1000, adt:192])
```

## 2. Build + train the model with a held-out batch

`held_out = "B3"` designates the query batch (leak-safe: its `cond_group` labels are masked). `donor_key` + `celltype_key` enable the donor predictor; `batch_key` is the site column whose effect the donor-level adversary removes; `target_condition` is the disease axis to predict. The donor-predictor knobs are the real engine hyperparameters used for the Liu figures.

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

``` r
preds <- m3_predict_donors(model)
cat("query donors:", nrow(preds), "\n")
#> query donors: 23
print(preds)
#>               donor is_reference predicted_label      prob_HC  prob_Severe
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

The query donors’ true `cond_group` lives in the metadata (never shown to the model). We join it back to score accuracy and ROC-AUC.

``` r
obs <- m3_dataset_obs(data)
truth <- unique(obs[, c("sample_id", "cond_group")])
truth <- stats::setNames(as.character(truth$cond_group), as.character(truth$sample_id))
preds$true_label <- truth[as.character(preds$donor)]
pos <- "Severe"
y_true  <- as.integer(preds$true_label == pos)
y_score <- preds[[paste0("prob_", pos)]]
acc <- mean(preds$predicted_label == preds$true_label)

auc_of <- function(score, label) {              # Mann-Whitney U / ROC-AUC
  r <- rank(score); n1 <- sum(label == 1); n0 <- sum(label == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  (sum(r[label == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
auc <- auc_of(y_score, y_true)
cat(sprintf("held-out accuracy = %.3f   ROC-AUC = %.3f\n", acc, auc))
#> held-out accuracy = 1.000   ROC-AUC = 1.000
```

## 5. Visualise — held-out ROC

``` r
roc_points <- function(score, label) {
  o <- order(score, decreasing = TRUE)
  tp <- cumsum(label[o] == 1); fp <- cumsum(label[o] == 0)
  data.frame(fpr = c(0, fp / max(fp, 1)), tpr = c(0, tp / max(tp, 1)))
}
rp <- roc_points(y_score, y_true)
ggplot(rp, aes(fpr, tpr)) +
  geom_abline(linetype = "dashed", colour = "grey") +
  geom_line(linewidth = 1, colour = "#4c72b0") +
  annotate("text", x = 0.65, y = 0.1, label = sprintf("AUC = %.3f", auc)) +
  labs(x = "False positive rate", y = "True positive rate", title = "Held-out donor ROC") +
  theme_classic()
```

<img src="../r_02_patient_prediction_media/8b3cce25a1d90c4bed0cc3c1aa1bfb8b881f11db.png" width="576" />

## 6. Patient-level embedding

`m3_donor_embedding()` returns the **patient-level (donor) embedding** — the corrected donor vector the model actually classifies (one row per donor). We UMAP it (same `umap-learn`, `random_state = 0`, as the Python tutorial) and colour by reference/query, true phenotype, and whether the held-out prediction was correct.

``` r
demb <- m3_donor_embedding(model)
info <- m3_predict_donors(model, include_reference = TRUE)
info <- info[match(demb$donor, info$donor), ]
info$true_label <- truth[as.character(demb$donor)]
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

**Done.** Leak-safe leave-one-batch-out donor disease prediction with a held-out ROC and a patient-level embedding UMAP.
