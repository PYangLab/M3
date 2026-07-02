# User-facing dataset constructors. A `m3_dataset` is a light S3 handle to a live
# Python m3.Dataset held in the worker (it keeps big raw counts off the R heap).

#' @keywords internal
new_m3_dataset <- function(handle, info, device = "auto") {
  structure(list(handle = handle, info = info, device = device),
            class = "m3_dataset")
}

#' @keywords internal
.dataset_info <- function(handle) .m3_call("dataset_info", h = handle)

#' Build an m3 dataset from a SingleCellExperiment / MultiAssayExperiment / list.
#'
#' RNA raw counts in the main assay, ADT/ATAC as altExps; the role columns
#' (condition / cell type / donor / batch) live in \code{colData}. A single
#' batch label is stamped on \code{obs$batch}; use \code{\link{m3_concat}} to
#' combine batches.
#'
#' @param x a \link[SingleCellExperiment]{SingleCellExperiment},
#'   \link[MultiAssayExperiment]{MultiAssayExperiment}, Seurat object, or a list
#'   with elements \code{rna}/\code{adt}/\code{atac} (features x cells) + \code{obs}.
#' @param batch batch label written to \code{obs$batch}.
#' @param assay,adt_exp,atac_exp assay / altExp selectors for SCE input.
#' @param device one of "auto", "cpu", "cuda" (session device, set on first use).
#' @return an \code{m3_dataset}.
#' @examples
#' \donttest{
#'   sce <- m3_example_sce()
#'   data <- m3_dataset(sce, batch = "batch1")
#'   data
#' }
#' @export
m3_dataset <- function(x, batch = "batch0", assay = "counts",
                       adt_exp = "ADT", atac_exp = "ATAC",
                       device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  parts <- .as_m3_parts(x, assay = assay, adt_exp = adt_exp, atac_exp = atac_exp)
  h <- .m3_call("dataset_from_parts", counts = parts$counts, obs = parts$obs,
                var = parts$var, batch = batch, .device = device)
  new_m3_dataset(h, .dataset_info(h), device)
}

#' Read one batch from paper-format HDF5 matrices + a metadata CSV.
#'
#' Mirrors \code{m3.read_h5} in Python: each matrix is an HDF5 file with the
#' counts under \code{matrix/data} and names under \code{matrix/features}; the
#' metadata CSV provides the per-cell role columns. Big data stays in Python
#' (never materialised on the R heap).
#'
#' @param rna,adt,atac paths to per-modality \code{.h5} files (adt/atac optional).
#' @param metadata path to the per-cell metadata \code{.csv}.
#' @param batch batch label written to \code{obs$batch}.
#' @param device session device (see \code{\link{m3_dataset}}).
#' @return an \code{m3_dataset}.
#' @examples
#' \donttest{
#'   d <- m3_read_h5(rna = "rna1.h5", adt = "adt1.h5",
#'                   metadata = "metadata1.csv", batch = "batch1")
#' }
#' @export
m3_read_h5 <- function(rna = NULL, adt = NULL, atac = NULL, metadata,
                       batch = "batch0", device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  abs <- function(p) if (is.null(p)) NULL else normalizePath(p, mustWork = TRUE)
  h <- .m3_call("dataset_read_h5", rna = abs(rna), adt = abs(adt), atac = abs(atac),
                metadata = abs(metadata), batch = batch, .device = device)
  new_m3_dataset(h, .dataset_info(h), device)
}

#' Read one batch from an AnnData \code{.h5ad} file.
#'
#' @param path path to the \code{.h5ad}.
#' @param batch batch label.
#' @param modality modality name used when \code{var} has no \code{feature_types}.
#' @param device session device.
#' @return an \code{m3_dataset}.
#' @export
m3_read_h5ad <- function(path, batch = "batch0", modality = "rna",
                         device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  h <- .m3_call("dataset_read_h5ad", path = normalizePath(path, mustWork = TRUE),
                batch = batch, modality = modality, .device = device)
  new_m3_dataset(h, .dataset_info(h), device)
}

#' Combine single-batch datasets into one multi-batch dataset.
#'
#' Features per modality are harmonised by name intersection (first batch's
#' order); batch labels must be unique. Mirrors \code{m3.concat}.
#'
#' @param datasets a list of \code{m3_dataset}s.
#' @return an \code{m3_dataset}.
#' @examples
#' \donttest{
#'   data <- m3_concat(list(d1, d2, d3))
#' }
#' @export
m3_concat <- function(datasets) {
  if (!length(datasets)) stop("m3_concat() needs at least one dataset.")
  if (length(datasets) == 1L) return(datasets[[1]])
  handles <- lapply(datasets, function(d) {
    if (!inherits(d, "m3_dataset")) stop("m3_concat() takes a list of m3_dataset objects.")
    d$handle
  })
  dev <- datasets[[1]]$device
  h <- .m3_call("dataset_concat", handles = handles, .device = dev)
  new_m3_dataset(h, .dataset_info(h), dev)
}

#' Load the built-in Liu et al. CITE-seq demo dataset.
#'
#' A stratified subsample of the Liu COVID-19 CITE-seq batches (RNA HVGs + ADT,
#' 3 batches) shipped with the package, ready for \code{\link{m3_train}}. The
#' same \code{liu_demo} object the Python package uses, so demo results match.
#'
#' @param device session device.
#' @return an \code{m3_dataset}.
#' @examples
#' \donttest{
#'   data <- m3_demo()
#'   data
#' }
#' @export
m3_demo <- function(device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  h <- .m3_call("dataset_demo", .device = device)
  new_m3_dataset(h, .dataset_info(h), device)
}

#' Per-cell metadata (obs) of a dataset as a data.frame.
#' @param dataset an \code{m3_dataset}.
#' @return a data.frame with one row per cell.
#' @export
m3_dataset_obs <- function(dataset) {
  stopifnot(inherits(dataset, "m3_dataset"))
  .m3_call("dataset_obs", h = dataset$handle, .device = dataset$device)
}

#' Dense count matrix (cells x features) for one modality of a dataset.
#' @param dataset an \code{m3_dataset}.
#' @param modality one of the dataset's modalities (e.g. "rna").
#' @return a numeric matrix, cells x features.
#' @export
m3_dataset_matrix <- function(dataset, modality) {
  stopifnot(inherits(dataset, "m3_dataset"))
  .m3_call("dataset_obs_matrix", h = dataset$handle, modality = modality,
           .device = dataset$device)
}

#' @export
print.m3_dataset <- function(x, ...) {
  i <- x$info
  mods <- paste(vapply(i$modalities, function(m) paste0(m, ":", i$n_features[[m]]), ""),
                collapse = ", ")
  cat("m3_dataset(n_cells=", i$n_cells, ", batches=[",
      paste(i$batches, collapse = ", "), "], modalities=[", mods, "])\n", sep = "")
  invisible(x)
}
