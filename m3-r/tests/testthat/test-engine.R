# Gated end-to-end test through basilisk (opt-in: M3_TEST_ENGINE=1).

test_that("full pipeline runs on the example SCE (CPU)", {
  skip_if_no_engine()
  on.exit(m3_shutdown(), add = TRUE)

  sce  <- m3_example_sce(n_cells = 160L, seed = 1L)   # spans batches B1/B2
  d1   <- m3_dataset(sce, device = "cpu")
  expect_s3_class(d1, "m3_dataset")
  expect_equal(d1$info$n_cells, 160L)

  # representation learning
  m1  <- m3_train(d1, condition_keys = c("cond_group", "Age_interval"),
                  celltype_key = "mergedcelltype", embedding_dim = 16L,
                  donor_prediction = FALSE, max_epochs = 2L, seed = 0L, device = "cpu")
  emb <- m3_embedding(m1, "bio")
  expect_equal(nrow(emb), 160L)
  expect_true(is.matrix(m3_reconstruct(m1)$rna))
  expect_equal(nrow(m3_cell_metadata(m1)), 160L)

  # donor prediction with a held-out batch (the SCE already spans B1/B2)
  m2  <- m3_train(d1, condition_keys = c("cond_group", "Age_interval"),
                  target_condition = "cond_group", celltype_key = "mergedcelltype",
                  donor_key = "sample_id", held_out = "B2",
                  embedding_dim = 16L, max_epochs = 2L,
                  donor_predictor = list(n_epochs = 3L), seed = 0L, device = "cpu")
  expect_true(m3_capabilities(m2)["predict_donors"])
  preds <- m3_predict_donors(m2)
  expect_true(all(c("donor", "predicted_label") %in% colnames(preds)))
  expect_gt(nrow(preds), 0L)            # B2 donors are disjoint from B1

  # attribution + augmentation
  attr <- m3_attribute(m2, reference_labels = "HC", n_steps = 6L)
  expect_s3_class(attr, "m3_attribution")
  expect_true(nrow(m3_top_celltypes(attr, min_cells_per_condition = 0L)) > 0L)
  aug <- m3_augment(m2, conditions = c("HC", "Severe"), n_donors = c(2L, 2L))
  expect_true(is.matrix(aug$expression$rna))
  expect_true(is.matrix(m3_generate(m2)$rna))
})

test_that("reproducible: same seed -> identical embedding", {
  skip_if_no_engine()
  on.exit(m3_shutdown(), add = TRUE)
  d <- m3_dataset(m3_example_sce(120L, seed = 7L), device = "cpu")
  e1 <- m3_embedding(m3_train(d, condition_keys = "cond_group", celltype_key = "mergedcelltype",
                              embedding_dim = 12L, donor_prediction = FALSE,
                              max_epochs = 3L, seed = 0L, device = "cpu"), "bio")
  e2 <- m3_embedding(m3_train(d, condition_keys = "cond_group", celltype_key = "mergedcelltype",
                              embedding_dim = 12L, donor_prediction = FALSE,
                              max_epochs = 3L, seed = 0L, device = "cpu"), "bio")
  expect_equal(max(abs(e1 - e2)), 0)
})

test_that("held_out_samples: hold out donors across batches", {
  skip_if_no_engine()
  on.exit(m3_shutdown(), add = TRUE)
  d <- m3_dataset(m3_example_sce(200L, seed = 3L), device = "cpu")
  obs <- m3_dataset_obs(d)
  held <- c(obs$sample_id[obs$batch == "B1"][1], obs$sample_id[obs$batch == "B2"][1])
  m <- m3_train(d, condition_keys = c("cond_group", "Age_interval"),
                target_condition = "cond_group", celltype_key = "mergedcelltype",
                donor_key = "sample_id", held_out_samples = held,
                embedding_dim = 12L, max_epochs = 2L,
                donor_predictor = list(n_epochs = 3L), seed = 0L, device = "cpu")
  preds <- m3_predict_donors(m)
  expect_setequal(as.character(preds$donor), as.character(held))   # query == the held-out donors
})
