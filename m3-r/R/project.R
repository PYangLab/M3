# Projection helpers that run the SAME Python implementations the m3 tutorials
# use (scanpy UMAP / umap-learn / sklearn PCA), so an R plot of their output has
# byte-identical coordinates to the Python tutorial -- no uwot-vs-umap-learn
# rotation. These need a started worker; pass the model/dataset device.

#' UMAP projection identical to the m3 Python tutorials.
#'
#' Runs scanpy's neighbors+UMAP (\code{method = "scanpy"}, as in Tutorial 1) or
#' \code{umap.UMAP} (\code{method = "umap"}, as in Tutorials 2 and 4) inside the
#' engine environment, so the coordinates match the Python tutorial exactly.
#'
#' @param x a numeric matrix (rows = points), e.g. an \code{\link{m3_embedding}}.
#' @param method "scanpy" or "umap".
#' @param n_neighbors UMAP neighbours (default 15; capped at nrow-1).
#' @param min_dist,spread,metric umap-learn parameters (\code{method = "umap"}).
#' @param random_state umap-learn seed (default 0).
#' @param device session device (defaults to "auto"; pass the model's device).
#' @return a 2-column numeric matrix of coordinates.
#' @examples
#' \donttest{
#'   emb <- m3_embedding(model, "bio")
#'   xy  <- m3_umap(emb, method = "scanpy")
#' }
#' @export
m3_umap <- function(x, method = c("scanpy", "umap"), n_neighbors = 15L,
                    min_dist = 0.1, spread = 1.0, metric = "euclidean",
                    random_state = 0L, device = "auto") {
  method <- match.arg(method)
  if (method == "scanpy") {
    .m3_call("umap_scanpy", X = x, n_neighbors = n_neighbors, .device = device)
  } else {
    .m3_call("umap_learn", X = x, n_neighbors = n_neighbors, min_dist = min_dist,
             spread = spread, metric = metric, random_state = random_state,
             .device = device)
  }
}

#' Two-component PCA (sklearn) — the small-sample fallback used in Tutorial 4.
#' @param x a numeric matrix (rows = points).
#' @param device session device.
#' @return a 2-column numeric matrix.
#' @export
m3_pca2 <- function(x, device = "auto") {
  .m3_call("pca2", X = x, .device = device)
}
