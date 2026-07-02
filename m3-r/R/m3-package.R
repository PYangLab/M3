#' m3: R interface to multimodal, multi-batch, condition-aware single-cell modelling
#'
#' Trains an integration variational autoencoder (RNA / ADT / ATAC) for
#' batch-corrected, condition-disentangled cell embeddings and, on top of it, an
#' adversarial donor-level disease predictor; explains predictions with
#' end-to-end integrated gradients; and augments batches with synthetic donors
#' and cells. The PyTorch implementation (the `m3-sc` Python package) is vendored
#' under \code{inst/python/m3} and driven via \pkg{reticulate} inside a
#' \pkg{basilisk} environment. Results are returned as R objects
#' (\code{m3_model}, \code{m3_attribution}, data frames, matrices).
#'
#' @section Workflow:
#' \code{\link{m3_demo}} / \code{\link{m3_read_h5}} / \code{\link{m3_dataset}} ->
#' \code{\link{m3_train}} -> \code{\link{m3_embedding}} /
#' \code{\link{m3_predict_donors}} / \code{\link{m3_attribute}} /
#' \code{\link{m3_augment}}.
#'
#' @keywords internal
#' @aliases m3-package
"_PACKAGE"
