# The S3 model handle: an integer pointing at the live Python m3.M3 in the
# worker, plus the converted contract/capability metadata for printing + dispatch.

#' @keywords internal
new_m3_model <- function(meta, device = "auto") {
  structure(c(meta, list(device = device)), class = "m3_model")
}

#' @keywords internal
.model_handle <- function(model) {
  if (inherits(model, "m3_model")) return(model$handle)
  if (methods::is(model, "SummarizedExperiment") ||
      methods::is(model, "MultiAssayExperiment")) {
    m <- S4Vectors::metadata(model)$m3
    if (!is.null(m)) return(m$handle)
  }
  stop("not a trained m3 model; train first with m3_train().")
}

#' @keywords internal
.resolve_m3 <- function(model) {
  if (inherits(model, "m3_model")) return(model)
  if (methods::is(model, "SummarizedExperiment") ||
      methods::is(model, "MultiAssayExperiment")) {
    m <- S4Vectors::metadata(model)$m3
    if (!is.null(m)) return(m)
  }
  stop("not a trained m3 model; train first with m3_train(), or pass an m3_model.")
}

#' Capabilities of a trained model (embedding / reconstruct / predict_donors).
#' @param model an \code{m3_model}.
#' @return a named logical vector.
#' @export
m3_capabilities <- function(model) {
  m <- .resolve_m3(model)
  unlist(.m3_call("model_capabilities", h = m$handle, .device = m$device))
}

#' Leak-safe reference vocabulary the model trained on (query labels excluded).
#' @param model an \code{m3_model}.
#' @return a named list (one entry per condition key).
#' @export
m3_reference_vocab <- function(model) {
  .resolve_m3(model)$reference_vocab
}

#' @export
print.m3_model <- function(x, ...) {
  caps <- names(Filter(isTRUE, x$capabilities))
  cat("m3_model\n")
  cat("  modalities     : ", paste(x$modalities, collapse = ", "),
      " (embedding_dim=", x$embedding_dim, ")\n", sep = "")
  cat("  condition keys : ", paste(x$condition_keys, collapse = ", "),
      "  (target: ", x$target_condition, ")\n", sep = "")
  cat("  roles          : ",
      "celltype=", if (is.null(x$celltype_key)) "-" else x$celltype_key,
      "  donor=", if (is.null(x$donor_key)) "-" else x$donor_key,
      "  batch=", if (is.null(x$batch_key)) "-" else x$batch_key, "\n", sep = "")
  if (length(x$held_out)) cat("  held out       : ", paste(x$held_out, collapse = ", "), "\n", sep = "")
  cat("  capabilities   : ", paste(caps, collapse = ", "), "\n", sep = "")
  invisible(x)
}
