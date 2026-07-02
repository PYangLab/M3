# A tiny synthetic multimodal SCE for examples, tests and CPU smoke runs.

#' A small multimodal SingleCellExperiment for examples and quick trials.
#'
#' Builds a synthetic RNA + ADT \link[SingleCellExperiment]{SingleCellExperiment}
#' with the role columns m3 expects: \code{mergedcelltype}, \code{cond_group}
#' (HC/Severe), \code{Age_interval}, \code{sample_id} (donor), and \code{batch}
#' (two batches \code{B1}/\code{B2}). Suitable for trying the workflow on CPU.
#'
#' @param n_cells number of cells (default 180).
#' @param seed RNG seed for the synthetic counts/labels (default 1).
#' @return a SingleCellExperiment with an "ADT" altExp and role columns in colData.
#' @examples
#' sce <- m3_example_sce()
#' sce
#' @export
m3_example_sce <- function(n_cells = 180L, seed = 1L) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("m3_example_sce() needs SingleCellExperiment.")
  }
  old <- .Random.seed_state(); on.exit(.Random.seed_restore(old), add = TRUE)
  set.seed(seed)
  celltypes <- c("CD14_Mono", "CD8_Mem", "NK", "B_Naive")
  conds <- c("HC", "Severe")
  batches <- c("B1", "B2")
  # 8 donors: 2 conditions x 2 batches x 2 donors
  donors <- expand.grid(cond = conds, batch = batches, rep = 1:2, stringsAsFactors = FALSE)
  # donor ids are namespaced by seed so distinct m3_example_sce() calls have
  # disjoint donors. One SCE already spans two batches (B1/B2), so a held-out
  # donor-prediction example can hold out a batch directly (held_out = "B2").
  donors$id <- sprintf("s%d_donor%02d", seed, seq_len(nrow(donors)))
  cell_donor <- sample(seq_len(nrow(donors)), n_cells, replace = TRUE)
  ct <- sample(celltypes, n_cells, replace = TRUE)

  mk <- function(nf, lam, pre) {
    m <- matrix(stats::rpois(nf * n_cells, lam), nf, n_cells,
                dimnames = list(paste0(pre, seq_len(nf)), paste0("cell", seq_len(n_cells))))
    m
  }
  rna <- mk(60, 5, "gene")
  adt <- mk(12, 8, "adt")
  # a faint disease signal so a tiny model has something to learn
  severe <- donors$cond[cell_donor] == "Severe"
  rna[1:5, severe] <- rna[1:5, severe] + 4L

  sce <- SingleCellExperiment::SingleCellExperiment(assays = list(counts = rna))
  SingleCellExperiment::altExp(sce, "ADT") <-
    SummarizedExperiment::SummarizedExperiment(list(counts = adt))
  cd <- SummarizedExperiment::colData(sce)
  cd$mergedcelltype <- ct
  cd$cond_group     <- donors$cond[cell_donor]
  cd$Age_interval   <- ifelse(donors$batch[cell_donor] == "B1", "young", "old")
  cd$sample_id      <- donors$id[cell_donor]
  cd$batch          <- donors$batch[cell_donor]
  SummarizedExperiment::colData(sce) <- cd
  sce
}

#' @keywords internal
.Random.seed_state <- function() {
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    get(".Random.seed", envir = globalenv(), inherits = FALSE)
  } else NULL
}

#' @keywords internal
.Random.seed_restore <- function(state) {
  if (is.null(state)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
  } else {
    assign(".Random.seed", state, envir = globalenv())
  }
}
