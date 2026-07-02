# Offline tests (no Python): S3 object constructors + print methods + dispatch.

test_that("print.m3_dataset is informative", {
  d <- new_m3_dataset(0L, list(n_cells = 100L, batches = c("a", "b"),
                               modalities = c("rna", "adt"),
                               n_features = list(rna = 1000L, adt = 192L),
                               obs_columns = "batch"))
  expect_s3_class(d, "m3_dataset")
  expect_output(print(d), "n_cells=100")
  expect_output(print(d), "rna:1000")
})

test_that("print.m3_model shows roles + capabilities", {
  m <- new_m3_model(list(handle = 0L, modalities = "rna", embedding_dim = 30L,
                         condition_keys = c("cond_group", "Age_interval"),
                         target_condition = "cond_group", celltype_key = "ct",
                         donor_key = "sid", batch_key = "B", held_out = "B3",
                         capabilities = list(embedding = TRUE, reconstruct = TRUE,
                                             predict_donors = TRUE),
                         reference_vocab = list(cond_group = c("HC", "Severe"))))
  expect_s3_class(m, "m3_model")
  expect_output(print(m), "predict_donors")
  expect_output(print(m), "B3")
  expect_equal(.model_handle(m), 0L)
  expect_equal(m3_reference_vocab(m)$cond_group, c("HC", "Severe"))
})

test_that("print.m3_attribution summarises tables", {
  a <- new_m3_attribution(0L, list(
    target_label = "Severe",
    genes = data.frame(feature = letters[1:3], importance = 3:1),
    celltypes = data.frame(celltype = c("x", "y"), importance = 2:1),
    donors = data.frame(donor = "d1", attribution = 1),
    feature_names = letters[1:3], celltype_names = c("x", "y")))
  expect_s3_class(a, "m3_attribution")
  expect_output(print(a), "target=Severe")
  expect_output(print(a), "genes=3")
})

test_that(".resolve_m3 errors on an untrained object", {
  expect_error(.resolve_m3(42), "train first")
})
