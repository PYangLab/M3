# Integrated-gradients attribution + ranked tables.

#' @keywords internal
new_m3_attribution <- function(handle, tables, device = "auto") {
  structure(list(handle = handle, target_label = tables$target_label,
                 genes = tables$genes, celltypes = tables$celltypes,
                 donors = tables$donors,
                 feature_names = tables$feature_names,
                 celltype_names = tables$celltype_names,
                 device = device),
            class = "m3_attribution")
}

#' End-to-end integrated-gradients attribution of the donor-level prediction.
#'
#' Attributes the \code{target_condition} prediction back to genes/proteins, cell
#' types and donors. \code{reference_labels} (the healthy/baseline label(s), e.g.
#' \code{"HC"}) define the IG baseline; without them the engine falls back to a
#' zero baseline that inflates housekeeping genes.
#'
#' @param model an \code{m3_model} trained with a donor predictor.
#' @param reference_labels baseline label(s) of \code{target_condition} (e.g. \code{"HC"}).
#' @param target_class optional explicit target class index (default: the first
#'   non-reference class).
#' @param n_steps integrated-gradients steps (default 50).
#' @return an \code{m3_attribution} with \code{$genes}, \code{$celltypes},
#'   \code{$donors}, ranked tables; pass it to \code{\link{m3_top_genes}} /
#'   \code{\link{m3_top_celltypes}} / \code{\link{m3_attribution_matrix}}.
#' @examples
#' \donttest{
#'   attr <- m3_attribute(model, reference_labels = "HC")
#'   head(attr$celltypes)
#' }
#' @export
m3_attribute <- function(model, reference_labels, target_class = NULL, n_steps = 50L) {
  m <- .resolve_m3(model)
  h <- .m3_call("attribute", h = m$handle,
                reference_labels = as.list(reference_labels),
                target_class = target_class, n_steps = n_steps, .device = m$device)
  tables <- .m3_call("attr_tables", h = h, .device = m$device)
  new_m3_attribution(h, tables, m$device)
}

#' Per-celltype-balanced top genes (the publication recipe).
#'
#' Drops cell types with fewer than \code{min_cells_per_condition} cells in
#' either condition, scores each gene by \code{mean(|gene x celltype IG|)} over
#' the kept cell types, excludes housekeeping/ribosomal genes by name, and
#' optionally restricts to one modality.
#'
#' @param attribution an \code{m3_attribution} from \code{\link{m3_attribute}}.
#' @param n number of genes to return (default 100).
#' @param min_cells_per_condition cell-type filter threshold (default 200; 0 to skip).
#' @param exclude_regex regex of names to drop; \code{NULL} to skip; missing uses
#'   the default housekeeping pattern.
#' @param modality "rna"/"adt"/"atac" to restrict ranking; \code{NULL} for all.
#' @return a data.frame: \code{feature}, \code{modality}, \code{score}, \code{n_celltypes_used}.
#' @export
m3_top_genes <- function(attribution, n = 100L, min_cells_per_condition = 200L,
                         exclude_regex, modality = NULL) {
  stopifnot(inherits(attribution, "m3_attribution"))
  rx <- if (missing(exclude_regex)) "__default__"
        else if (is.null(exclude_regex)) "" else exclude_regex
  .m3_call("attr_top_genes", h = attribution$handle, n = n,
           min_cells_per_condition = min_cells_per_condition,
           exclude_regex = rx, modality = modality, .device = attribution$device)
}

#' Cell-type importance ranking, filtered to types with enough cells.
#'
#' @param attribution an \code{m3_attribution}.
#' @param min_cells_per_condition keep cell types with at least this many cells in
#'   both conditions (default 200; 0 returns the raw table).
#' @return a data.frame: \code{celltype}, \code{importance}.
#' @export
m3_top_celltypes <- function(attribution, min_cells_per_condition = 200L) {
  stopifnot(inherits(attribution, "m3_attribution"))
  .m3_call("attr_top_celltypes", h = attribution$handle,
           min_cells_per_condition = min_cells_per_condition, .device = attribution$device)
}

#' The full cell x gene signed attribution matrix.
#' @param attribution an \code{m3_attribution}.
#' @return a numeric matrix, cells x features.
#' @export
m3_attribution_matrix <- function(attribution) {
  stopifnot(inherits(attribution, "m3_attribution"))
  .m3_call("attr_matrix", h = attribution$handle, .device = attribution$device)
}

#' The signed per-(cell-type, gene) attribution matrix.
#' @param attribution an \code{m3_attribution}.
#' @return a numeric matrix, cell types x features (rows = \code{attribution$celltype_names}).
#' @export
m3_gene_celltype_matrix <- function(attribution) {
  stopifnot(inherits(attribution, "m3_attribution"))
  .m3_call("attr_gene_celltype_matrix", h = attribution$handle, .device = attribution$device)
}

#' @export
print.m3_attribution <- function(x, ...) {
  nd <- if (is.null(x$donors)) 0L else nrow(x$donors)
  cat("m3_attribution(target=", if (is.null(x$target_label)) "?" else x$target_label,
      ", genes=", nrow(x$genes), ", celltypes=", nrow(x$celltypes),
      ", donors=", nd, ")\n", sep = "")
  invisible(x)
}
