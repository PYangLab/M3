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
  expect_true(all(c("sample_id", "predicted_label") %in% colnames(preds)))
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
  # ... and the seed is the thing doing it: a different seed is a different run.
  e3 <- m3_embedding(m3_train(d, condition_keys = "cond_group", celltype_key = "mergedcelltype",
                              embedding_dim = 12L, donor_prediction = FALSE,
                              max_epochs = 3L, seed = 1L, device = "cpu"), "bio")
  expect_gt(max(abs(e1 - e3)), 0)
  expect_true("obs_names" %in% colnames(m3_cell_metadata(m3_train(
    d, condition_keys = "cond_group", celltype_key = "mergedcelltype", embedding_dim = 12L,
    donor_prediction = FALSE, max_epochs = 1L, device = "cpu"))))
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
  expect_setequal(as.character(preds$sample_id), as.character(held))   # query == the held-out donors
})

test_that("m3_demo(n_cells=) loads a smaller demo with equal cells per batch", {
  skip_if_no_engine()
  on.exit(m3_shutdown(), add = TRUE)
  d <- m3_demo(n_cells = 3000, device = "cpu")     # a plain R number, not 3000L
  expect_equal(d$info$n_cells, 3000L)
  expect_equal(as.vector(table(m3_dataset_obs(d)$batch)), c(1000L, 1000L, 1000L))
  expect_error(m3_demo(n_cells = 300, device = "cpu"), "n_cells must be between")
})

test_that("m3_save / m3_load: a loaded model gives the saved model's read-outs", {
  skip_if_no_engine()
  on.exit(m3_shutdown(), add = TRUE)
  d <- m3_dataset(m3_example_sce(200L, seed = 5L), device = "cpu")
  m <- m3_train(d, condition_keys = c("cond_group", "Age_interval"),
                target_condition = "cond_group", celltype_key = "mergedcelltype",
                donor_key = "sample_id", held_out = "B2",
                embedding_dim = 12L, max_epochs = 2L,
                donor_predictor = list(n_epochs = 3L), seed = 0L, device = "cpu")
  emb   <- m3_embedding(m, "bio")
  preds <- m3_predict_donors(m)
  # Column by column: a data.frame from the engine also carries a pointer to the
  # pandas index it came from, which differs between any two calls.
  same_columns <- function(a, b) {
    expect_identical(names(a), names(b))
    for (n in names(b)) expect_identical(a[[n]], b[[n]])
  }

  path <- m3_save(m, tempfile(fileext = ".m3"))
  expect_true(file.exists(path))
  l <- m3_load(path, d, device = "cpu")
  expect_s3_class(l, "m3_model")
  expect_identical(m3_capabilities(l), m3_capabilities(m))
  expect_identical(m3_embedding(l, "bio"), emb)
  same_columns(m3_predict_donors(l), preds)

  # the file outlives the session that trained the model
  m3_shutdown()
  d2 <- m3_dataset(m3_example_sce(200L, seed = 5L), device = "cpu")
  l2 <- m3_load(path, d2, device = "cpu")
  expect_identical(m3_embedding(l2, "bio"), emb)
  same_columns(m3_predict_donors(l2), preds)

  # a different dataset is refused
  other <- m3_dataset(m3_example_sce(200L, seed = 6L), device = "cpu")
  expect_error(m3_load(path, other, device = "cpu"), "not the dataset this model was trained on")
})

test_that("the batch comes from batch_key, and no batch is invented", {
  skip_if_no_engine()
  on.exit(m3_shutdown(), add = TRUE)

  sce  <- m3_example_sce(n_cells = 160L, seed = 1L)
  meta <- as.data.frame(SummarizedExperiment::colData(sce))
  names(meta)[names(meta) == "batch"] <- "run"
  parts <- list(rna = SingleCellExperiment::counts(sce), obs = meta)

  d <- m3_dataset(parts, device = "cpu")
  expect_false("batch" %in% colnames(m3_dataset_obs(d)))
  expect_output(print(d), "batches=not set", fixed = TRUE)

  expect_error(
    m3_train(d, condition_keys = "cond_group", celltype_key = "mergedcelltype",
             donor_prediction = FALSE, max_epochs = 1L, device = "cpu"),
    "batch_key=")
  m <- m3_train(d, condition_keys = "cond_group", celltype_key = "mergedcelltype",
                batch_key = "run", embedding_dim = 16L, donor_prediction = FALSE,
                max_epochs = 1L, device = "cpu")
  expect_setequal(unique(m3_cell_metadata(m)$run), c("B1", "B2"))

  # a label that contradicts an existing batch column is refused
  expect_error(m3_dataset(sce, batch = "other", device = "cpu"), "already has a 'batch' column")
})

test_that("the cell-level model trains without cell-type labels", {
  skip_if_no_engine()
  on.exit(m3_shutdown(), add = TRUE)

  sce  <- m3_example_sce(n_cells = 160L, seed = 1L)
  meta <- as.data.frame(SummarizedExperiment::colData(sce))[, c("batch", "cond_group")]
  d <- m3_dataset(list(rna = SingleCellExperiment::counts(sce), obs = meta), device = "cpu")

  m <- m3_train(d, condition_keys = "cond_group", embedding_dim = 16L,
                max_epochs = 1L, device = "cpu")
  expect_equal(nrow(m3_embedding(m, "bio")), 160L)
  expect_false(m3_capabilities(m)[["predict_donors"]])
  expect_false(any(startsWith(colnames(m3_cell_metadata(m)), "__m3")))
})

test_that("m3_read_h5ad takes one file per modality and matches cells by name", {
  skip_if_no_engine()
  on.exit(m3_shutdown(), add = TRUE)

  dir <- tempfile("m3_h5ad_")
  dir.create(dir)
  # write rna.h5ad (batches B1, B2) and adt.h5ad (B1 only, rows shuffled) with the
  # worker's own anndata, so the test needs no R-side h5ad writer
  code <- sprintf('
import anndata as ad, numpy as np, pandas as pd
names = [f"c{i}-{b}" for b in ("B1", "B2") for i in range(20)]
obs = pd.DataFrame({"batch": [n.split("-")[1] for n in names],
                    "cond": ["hc", "dis"] * 20}, index=names)
rna = ad.AnnData(np.arange(40 * 5, dtype=np.float32).reshape(40, 5), obs=obs,
                 var=pd.DataFrame(index=[f"g{j}" for j in range(5)]))
adt = rna[rna.obs["batch"] == "B1", :2].copy()
adt.var_names = ["p0", "p1"]
adt = adt[np.random.default_rng(0).permutation(adt.n_obs)].copy()
rna.write_h5ad(r"%s/rna.h5ad"); adt.write_h5ad(r"%s/adt.h5ad")
', dir, dir)
  .m3_proc("cpu")
  if (identical(.m3_runtime$mode, "reticulate")) {
    reticulate::py_run_string(code)
  } else {
    basilisk::basiliskRun(.m3_runtime$proc, function(code) {
      reticulate::py_run_string(code); NULL }, code = code)
  }

  d <- m3_read_h5ad(list(rna = file.path(dir, "rna.h5ad"), adt = file.path(dir, "adt.h5ad")),
                    device = "cpu")
  expect_output(print(d), "batches=[B1, B2]", fixed = TRUE)
  obs <- m3_dataset_obs(d)
  adt <- m3_dataset_matrix(d, "adt")
  i <- which(obs$obs_names == "c7-B1")
  expect_equal(unname(adt[i, ]), c(35, 36))               # paired by name, not by row
  expect_true(all(adt[obs$batch == "B2", ] == 0))

  expect_error(m3_read_h5ad(c(file.path(dir, "rna.h5ad"), file.path(dir, "adt.h5ad")),
                            device = "cpu"), "named list")
})

test_that("m3_umap runs both methods inside the R process", {
  skip_if_no_engine()
  on.exit(m3_shutdown(), add = TRUE)
  # On macOS the scanpy method segfaulted R under numba's default threading-layer
  # selection (see _r_api.py); a regression ends the test run, it does not fail quietly.
  x <- matrix(stats::rnorm(300 * 10), 300)
  for (method in c("scanpy", "umap")) {
    xy <- m3_umap(x, method = method, n_neighbors = 10L, device = "cpu")
    expect_equal(dim(xy), c(300L, 2L))
    expect_true(all(is.finite(xy)))
  }
})
