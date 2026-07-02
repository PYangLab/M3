# Generative augmentation: synthetic donors per condition, and resampled cells.

#' Synthesize new donors per condition.
#'
#' Resamples real donor templates per condition through the VAE posterior,
#' producing fresh synthetic donors with per-cell-type expression.
#'
#' @param model an \code{m3_model}.
#' @param conditions character vector of conditions to synthesize (in order).
#' @param n_donors integer vector, synthetic donors per condition (same length).
#' @param tau posterior temperature (default 0.8).
#' @param batch optional batch label to template from (needs \code{batch_key} at
#'   training); \code{NULL} draws from all batches.
#' @param seed integer seed (default 42).
#' @return a list with \code{expression} (named list of cells x features matrices,
#'   one per modality) and \code{obs} (a data.frame, one row per synthetic cell).
#' @examples
#' \donttest{
#'   aug <- m3_augment(model, conditions = c("HC", "Severe"), n_donors = c(3, 3), tau = 0.8)
#'   dim(aug$expression$rna)
#' }
#' @export
m3_augment <- function(model, conditions, n_donors, tau = 0.8, batch = NULL, seed = 42L) {
  m <- .resolve_m3(model)
  .m3_call("augment", h = m$handle, conditions = as.list(conditions),
           n_donors = as.list(as.integer(n_donors)), tau = tau, batch = batch,
           seed = seed, .device = m$device)
}

#' Posterior-resampled cells (1:1 with the reference cells).
#'
#' @param model an \code{m3_model}.
#' @param tau posterior temperature (default 0.8).
#' @param seed integer seed (default 42).
#' @return a named list of numeric matrices (one per modality), cells x features.
#' @export
m3_generate <- function(model, tau = 0.8, seed = 42L) {
  m <- .resolve_m3(model)
  .m3_call("generate", h = m$handle, tau = tau, seed = seed, .device = m$device)
}
