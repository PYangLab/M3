# Donor-level disease prediction readouts.

#' Donor-level disease prediction.
#'
#' Returns per-donor predicted label + class probabilities. By default only the
#' held-out query donors are returned (set \code{include_reference = TRUE} for all).
#'
#' @param model an \code{m3_model} trained with a donor predictor.
#' @param include_reference also return the reference donors (default FALSE).
#' @return a data.frame: \code{donor}, \code{is_reference}, \code{predicted_label},
#'   and one \code{prob_<label>} column per class.
#' @examples
#' \donttest{
#'   preds <- m3_predict_donors(model)
#' }
#' @export
m3_predict_donors <- function(model, include_reference = FALSE) {
  m <- .resolve_m3(model)
  .m3_call("predict_donors", h = m$handle, include_reference = include_reference,
           .device = m$device)
}

#' Patient/donor-level embedding (the corrected donor vectors classified).
#'
#' @param model an \code{m3_model} trained with a donor predictor.
#' @return a data.frame with a \code{donor} column, an \code{is_reference} flag,
#'   and the embedding dimensions (\code{m3_0}, \code{m3_1}, ...).
#' @export
m3_donor_embedding <- function(model) {
  m <- .resolve_m3(model)
  .m3_call("donor_embedding", h = m$handle, .device = m$device)
}
