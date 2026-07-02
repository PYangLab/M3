# Offline tests (no Python): the SCE -> m3 parts conversion.

test_that("m3_example_sce has the role columns m3 needs", {
  sce <- m3_example_sce(n_cells = 50L, seed = 1L)
  cd <- SummarizedExperiment::colData(sce)
  for (col in c("mergedcelltype", "cond_group", "Age_interval", "sample_id", "batch")) {
    expect_true(col %in% colnames(cd))
  }
  expect_true("ADT" %in% SingleCellExperiment::altExpNames(sce))
  expect_equal(ncol(sce), 50L)
})

test_that(".as_m3_parts orients counts cells x features and carries obs", {
  sce <- m3_example_sce(n_cells = 40L, seed = 2L)
  parts <- m3:::.as_m3_parts(sce)
  # RNA assay is features x cells (60 x 40); m3 wants cells x features (40 x 60)
  expect_equal(dim(parts$counts$rna), c(40L, 60L))
  expect_equal(dim(parts$counts$adt), c(40L, 12L))
  expect_equal(length(parts$var$rna), 60L)
  expect_equal(length(parts$var$adt), 12L)
  expect_equal(nrow(parts$obs), 40L)
  expect_true(all(c("cond_group", "mergedcelltype", "batch") %in% colnames(parts$obs)))
  # transpose is correct: parts$counts$rna[i, j] == assay[j, i]
  a <- as.matrix(SummarizedExperiment::assay(sce, "counts"))
  expect_equal(parts$counts$rna[3, 5], a[5, 3])
})

test_that(".as_m3_parts rejects unsupported input", {
  expect_error(m3:::.as_m3_parts(42), "Unsupported input")
})

test_that(".as_m3_parts handles a plain list with obs", {
  rna <- matrix(1:20, 4, 5, dimnames = list(paste0("g", 1:4), paste0("c", 1:5)))
  obs <- data.frame(batch = rep("a", 5), cond_group = rep("HC", 5))
  parts <- m3:::.as_m3_parts(list(rna = rna, obs = obs))
  expect_equal(dim(parts$counts$rna), c(5L, 4L))
  expect_equal(nrow(parts$obs), 5L)
})
