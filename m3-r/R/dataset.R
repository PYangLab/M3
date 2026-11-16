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
#' (condition / cell type / donor / batch) live in \code{colData}. The batches
#' are read from the metadata column named by \code{batch_key} in
#' \code{\link{m3_train}} (default \code{"batch"}); no batch is made up when
#' there is none. Use \code{\link{m3_concat}} to combine batches that were
#' built separately.
#'
#' @param x a \link[SingleCellExperiment]{SingleCellExperiment},
#'   \link[MultiAssayExperiment]{MultiAssayExperiment}, Seurat object, or a list
#'   with elements \code{rna}/\code{adt}/\code{atac} (features x cells) + \code{obs}.
#' @param batch when all cells are one batch, its label, written to
#'   \code{obs$batch}. Leave it \code{NULL} when the metadata has a column that
#'   identifies the batches. It is an error if the metadata already has a
#'   \code{batch} column with other values.
#' @param assay,adt_exp,atac_exp assay / altExp selectors for SCE input.
#' @param device one of "auto", "cpu", "cuda" (session device, set on first use).
#' @return an \code{m3_dataset}.
#' @examples
#' \donttest{
#'   sce <- m3_example_sce()
#'   data <- m3_dataset(sce)
#'   data
#' }
#' @export
m3_dataset <- function(x, batch = NULL, assay = "counts",
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
#' @param batch the label of this batch, written to \code{obs$batch}; see
#'   \code{\link{m3_dataset}}.
#' @param device session device (see \code{\link{m3_dataset}}).
#' @return an \code{m3_dataset}.
#' @examples
#' \donttest{
#'   d <- m3_read_h5(rna = "rna1.h5", adt = "adt1.h5",
#'                   metadata = "metadata1.csv", batch = "batch1")
#' }
#' @export
m3_read_h5 <- function(rna = NULL, adt = NULL, atac = NULL, metadata,
                       batch = NULL, device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  abs <- function(p) if (is.null(p)) NULL else normalizePath(p, mustWork = TRUE)
  h <- .m3_call("dataset_read_h5", rna = abs(rna), adt = abs(adt), atac = abs(atac),
                metadata = abs(metadata), batch = batch, .device = device)
  new_m3_dataset(h, .dataset_info(h), device)
}

#' Read AnnData \code{.h5ad} file(s).
#'
#' Mirrors \code{m3.read_h5ad} in Python. Give one file, or one file per
#' modality as a named list, e.g. \code{list(rna = "rna.h5ad", adt = "adt.h5ad")}.
#' With one file per modality, the cells are matched by name and a cell missing
#' from a modality is marked as not measured for it. Each file's \code{obs}
#' needs the batch column (\code{batch_key}), and within a batch every modality
#' must have the same cells, or none of them; this is checked from \code{obs}
#' before any counts are loaded. Prints the number of cells per batch and
#' modality.
#'
#' @param path path to one \code{.h5ad}, or a named list of paths, one per
#'   modality.
#' @param batch when all cells are one batch, its label, written to
#'   \code{obs$batch}; see \code{\link{m3_dataset}}.
#' @param modality modality name used for a single file whose \code{var} has
#'   no \code{feature_types}.
#' @param batch_key with one file per modality, the \code{obs} column that
#'   identifies the batches.
#' @param layer where the raw counts are, if not in \code{X}: a layer name, or
#'   (with one file per modality) a named list of layer names.
#' @param join with one file per modality: \code{"exact"} (every modality of a
#'   batch has the same cells, else an error) or \code{"inner"} (keep the cells
#'   every modality of the batch has, and report the others).
#' @param device session device.
#' @return an \code{m3_dataset}.
#' @examples
#' \donttest{
#'   data <- m3_read_h5ad(list(rna = "rna.h5ad", adt = "adt.h5ad"))
#' }
#' @export
m3_read_h5ad <- function(path, batch = NULL, modality = "rna", batch_key = "batch",
                         layer = NULL, join = c("exact", "inner"),
                         device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  join <- match.arg(join)
  if (is.list(path) || length(path) > 1L || !is.null(names(path))) {
    if (is.null(names(path)) || any(!nzchar(names(path)))) {
      stop("give one path per modality as a named list, e.g. ",
           "list(rna = \"rna.h5ad\", adt = \"adt.h5ad\").")
    }
    path <- lapply(path, normalizePath, mustWork = TRUE)
  } else {
    path <- normalizePath(path, mustWork = TRUE)
  }
  if (!is.null(layer) && (is.list(layer) || !is.null(names(layer)))) layer <- as.list(layer)
  h <- .m3_call("dataset_read_h5ad", path = path, batch = batch, modality = modality,
                batch_key = batch_key, layer = layer, join = join, .device = device)
  new_m3_dataset(h, .dataset_info(h), device)
}

#' Combine datasets into one multi-batch dataset.
#'
#' Features per modality are harmonised by name intersection (first batch's
#' order). The datasets either all have a \code{batch} column, with labels
#' unique across inputs, or none has one; in the second case \code{batch_key}
#' in \code{\link{m3_train}} names the column that identifies the batches.
#' Mirrors \code{m3.concat}.
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
#' @param n_cells load a smaller, fixed subsample of this many cells instead of
#'   all 30,534, for a quicker run (training time is proportional to the number
#'   of cells). The same cells are returned on every call, and they are the cells
#'   Python's \code{m3.datasets.liu_demo(n_cells=)} returns. They come from
#'   seven well-separated major cell types that every batch has (B_Mem,
#'   CD4_Mem, CD8_Va7.2, Mono_Classical, Mono_Nonclassical, NK, Plasmablast).
#'   Each batch contributes equally and each of these types keeps at least 50
#'   cells in every batch. Must be between 1,050 and 13,329. \code{NULL} (the
#'   default) loads the whole demo, with all 17 cell types. Meant for cell-level integration: donor
#'   prediction and attribution need enough cells per donor, so use the whole
#'   demo there.
#' @return an \code{m3_dataset}.
#' @examples
#' \donttest{
#'   data <- m3_demo()
#'   data
#'   small <- m3_demo(n_cells = 6000)
#' }
#' @export
m3_demo <- function(device = c("auto", "cpu", "cuda"), n_cells = NULL) {
  device <- match.arg(device)
  if (!is.null(n_cells)) {
    if (!is.numeric(n_cells) || length(n_cells) != 1L || is.na(n_cells) ||
        n_cells != round(n_cells)) {
      stop("n_cells must be a single whole number.")
    }
    n_cells <- as.integer(n_cells)
  }
  h <- .m3_call("dataset_demo", n_cells = n_cells, .device = device)
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
  batches <- if (length(i$batches)) paste0("[", paste(i$batches, collapse = ", "), "]") else "not set"
  cat("m3_dataset(n_cells=", i$n_cells, ", batches=", batches,
      ", modalities=[", mods, "])\n", sep = "")
  invisible(x)
}
