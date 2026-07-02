# Turn supported R inputs into m3's per-modality cells x features parts.
#
# m3 expects RAW counts oriented cells x features, plus an obs table carrying the
# role columns (condition / celltype / donor / batch). SCE/MAE assays
# are features x cells, so we transpose. (Dense input is realised
# in memory; for large data prefer the file path via m3_read_h5().)

#' @keywords internal
.t_dense <- function(m) {
  # genes x cells (assay) -> cells x features (m3), realised dense.
  t(as.matrix(m))
}

#' Pull RNA/ADT/ATAC count matrices (cells x features), var names, and obs.
#'
#' SCE: RNA in assay `assay` (default "counts"); ADT/ATAC as altExps named
#' `adt_exp`/`atac_exp`. MAE: experiments named rna/adt/atac (column-aligned).
#' list: elements rna/adt/atac (features x cells) + obs.
#' @keywords internal
.as_m3_parts <- function(x, assay = "counts", adt_exp = "ADT", atac_exp = "ATAC") {
  if (inherits(x, "Seurat")) {
    if (!requireNamespace("Seurat", quietly = TRUE)) {
      stop("Seurat input requires the Seurat package.")
    }
    x <- Seurat::as.SingleCellExperiment(x)
  }

  counts <- list(); var <- list()
  if (methods::is(x, "SingleCellExperiment")) {
    rna <- SummarizedExperiment::assay(x, assay)
    counts$rna <- .t_dense(rna)
    var$rna <- rownames(rna)
    ae <- SingleCellExperiment::altExpNames(x)
    if (!is.null(adt_exp) && adt_exp %in% ae) {
      a <- SummarizedExperiment::assay(SingleCellExperiment::altExp(x, adt_exp), 1L)
      counts$adt <- .t_dense(a); var$adt <- rownames(a)
    }
    if (!is.null(atac_exp) && atac_exp %in% ae) {
      a <- SummarizedExperiment::assay(SingleCellExperiment::altExp(x, atac_exp), 1L)
      counts$atac <- .t_dense(a); var$atac <- rownames(a)
    }
    obs <- as.data.frame(SummarizedExperiment::colData(x), optional = TRUE)
  } else if (methods::is(x, "MultiAssayExperiment")) {
    if (!requireNamespace("MultiAssayExperiment", quietly = TRUE)) {
      stop("MultiAssayExperiment input requires the MultiAssayExperiment package.")
    }
    nm <- names(x)
    rna <- MultiAssayExperiment::assay(x, "rna")
    counts$rna <- .t_dense(rna); var$rna <- rownames(rna)
    if ("adt" %in% nm) {
      a <- MultiAssayExperiment::assay(x, "adt"); counts$adt <- .t_dense(a); var$adt <- rownames(a)
    }
    if ("atac" %in% nm) {
      a <- MultiAssayExperiment::assay(x, "atac"); counts$atac <- .t_dense(a); var$atac <- rownames(a)
    }
    obs <- as.data.frame(MultiAssayExperiment::colData(x), optional = TRUE)
  } else if (is.list(x) && !is.null(x$rna)) {
    counts$rna <- .t_dense(x$rna); var$rna <- rownames(as.matrix(x$rna))
    if (!is.null(x$adt))  { counts$adt  <- .t_dense(x$adt);  var$adt  <- rownames(as.matrix(x$adt)) }
    if (!is.null(x$atac)) { counts$atac <- .t_dense(x$atac); var$atac <- rownames(as.matrix(x$atac)) }
    if (is.null(x$obs)) stop("list input needs an `obs` data.frame element.")
    obs <- as.data.frame(x$obs)
  } else {
    stop("Unsupported input class: ", paste(class(x), collapse = "/"))
  }

  n <- nrow(counts$rna)
  for (m in names(var)) {
    if (is.null(var[[m]])) var[[m]] <- paste0(m, "_", seq_len(ncol(counts[[m]])))
  }
  if (nrow(obs) != n) {
    stop("obs / colData has ", nrow(obs), " rows but the matrix has ", n, " cells.")
  }
  obs <- as.data.frame(lapply(obs, function(col) {
    if (is.factor(col)) as.character(col) else col
  }), stringsAsFactors = FALSE, check.names = FALSE)
  list(counts = counts, var = var, obs = obs)
}
