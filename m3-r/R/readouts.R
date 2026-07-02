# Cell-level readouts off a trained model.

#' Cell-level latent embedding.
#'
#' @param model an \code{m3_model}.
#' @param part one of "bio" (intrinsic + condition latents, batch removed),
#'   "intrinsic", "batch", or one of the model's condition keys.
#' @return a numeric matrix, cells x latent dimensions (row order matches
#'   \code{\link{m3_cell_metadata}}).
#' @examples
#' \donttest{
#'   emb <- m3_embedding(model, part = "bio")
#' }
#' @export
m3_embedding <- function(model, part = "bio") {
  m <- .resolve_m3(model)
  .m3_call("embedding", h = m$handle, part = part, .device = m$device)
}

#' Batch-corrected per-modality reconstruction (posterior-mean decode).
#'
#' @param model an \code{m3_model}.
#' @param remove_batch zero the batch latent before decoding (default TRUE).
#' @return a named list of numeric matrices (one per modality), cells x features.
#' @export
m3_reconstruct <- function(model, remove_batch = TRUE) {
  m <- .resolve_m3(model)
  .m3_call("reconstruct", h = m$handle, remove_batch = remove_batch, .device = m$device)
}

#' Row-aligned cell metadata for the embedding / reconstruction rows.
#'
#' @param model an \code{m3_model}.
#' @return a data.frame (reference rows first, then any held-out query rows).
#' @export
m3_cell_metadata <- function(model) {
  m <- .resolve_m3(model)
  .m3_call("cell_metadata", h = m$handle, .device = m$device)
}
