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

#' Save a trained model to a file.
#'
#' Writes the trained weights and the model's settings to \code{path}, to be read
#' back with \code{\link{m3_load}} in a later session. The file does not hold the
#' data: loading needs the dataset the model was trained on.
#'
#' @param model an \code{m3_model}.
#' @param path file to write (for example \code{"model.m3"}).
#' @return \code{path}, invisibly.
#' @examples
#' \donttest{
#'   data  <- m3_dataset(m3_example_sce())
#'   model <- m3_train(data, condition_keys = c("cond_group", "Age_interval"),
#'                     celltype_key = "mergedcelltype", donor_prediction = FALSE,
#'                     max_epochs = 2)
#'   path  <- m3_save(model, tempfile(fileext = ".m3"))
#'   again <- m3_load(path, data)
#' }
#' @export
m3_save <- function(model, path) {
  m <- .resolve_m3(model)
  path <- normalizePath(path, mustWork = FALSE)
  .m3_call("model_save", h = m$handle, path = path, .device = m$device)
  invisible(path)
}

#' Load a model saved with \code{m3_save}.
#'
#' The loaded model gives the read-outs of the model that was saved
#' (\code{\link{m3_embedding}}, \code{\link{m3_predict_donors}},
#' \code{\link{m3_attribute}}, ...) without training again. \code{data} must be
#' the dataset the model was trained on: the same cells in the same order, with
#' the same counts, feature names and role columns. Anything else is refused, as
#' is a damaged file.
#'
#' @param path a file written by \code{\link{m3_save}}, or one of the models
#'   shipped with the package (\code{\link{m3_demo_model}}).
#' @param data the \code{\link{m3_dataset}} the model was trained on, or any
#'   input \code{\link{m3_dataset}} accepts.
#' @param device one of "auto", "cpu", "cuda" (session device, fixed on first use).
#' @return an \code{m3_model}.
#' @examples
#' \donttest{
#'   data  <- m3_demo()
#'   model <- m3_load(m3_demo_model("patient_prediction"), data)
#'   m3_predict_donors(model)
#' }
#' @export
m3_load <- function(path, data, device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  if (!inherits(data, "m3_dataset")) {
    data <- m3_dataset(data, device = device)
  }
  path <- normalizePath(path, mustWork = TRUE)
  meta <- .m3_call("model_load", path = path, dataset_handle = data$handle,
                   .device = data$device)
  new_m3_model(meta, data$device)
}

#' Path of a model trained in advance on the full demo data.
#'
#' Training on the full \code{\link{m3_demo}} data takes a while; these files let
#' you go straight to the read-outs with \code{\link{m3_load}}.
#'
#' @param name \code{"patient_prediction"} (batch \code{B3} held out, as in the
#'   donor-level prediction tutorial) or \code{"attribution"} (trained on all
#'   three batches, as in the feature attribution tutorial).
#' @param device one of "auto", "cpu", "cuda" (session device, fixed on first use).
#' @return the path of the model file inside the installed package.
#' @examples
#' \donttest{
#'   m3_demo_model("attribution")
#' }
#' @export
m3_demo_model <- function(name = c("patient_prediction", "attribution"),
                          device = c("auto", "cpu", "cuda")) {
  name <- match.arg(name)
  device <- match.arg(device)
  .m3_call("demo_model", name = name, .device = device)
}
