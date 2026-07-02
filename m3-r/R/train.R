# Training: build (if needed) a dataset, construct + train m3.M3 in the worker.

#' Train an m3 model.
#'
#' Trains the Stage-1 integration variational autoencoder and, when
#' \code{donor_key} + \code{celltype_key} are given (and donor prediction is not
#' switched off), the Stage-2 adversarial donor-level disease predictor on top.
#' With \code{held_out} the held-out batch is leak-safe: its
#' \code{target_condition} labels are masked during training and predicted by
#' \code{\link{m3_predict_donors}}.
#'
#' @param data an \code{\link{m3_dataset}}, or any input \code{\link{m3_dataset}}
#'   accepts (SCE / MAE / Seurat / list), converted automatically.
#' @param condition_keys character vector of condition columns in \code{obs}
#'   (m3 is condition-aware; at least one is required).
#' @param target_condition the condition to predict / attribute (default: the
#'   first of \code{condition_keys}).
#' @param celltype_key,donor_key,batch_key role columns in \code{obs}. The donor
#'   predictor + attribution need \code{donor_key} + \code{celltype_key};
#'   \code{batch_key} (default \code{"batch"}) is the obs column identifying
#'   batches — both the held-out/integration unit and the site the donor
#'   adversary removes.
#' @param held_out character vector of batch labels to hold out (leak-safe).
#' @param held_out_samples character vector of donor IDs (values of
#'   \code{donor_key}) to hold out as the query set — they may span multiple
#'   batches (the donor adversary still removes batch). Mutually exclusive with
#'   \code{held_out}. Their \code{target_condition} labels are not required
#'   (leave them \code{NA}); they are masked in training and recovered by
#'   \code{\link{m3_predict_donors}}.
#' @param hvg optional named list of per-modality HVG counts, e.g. \code{list(rna = 1000)}.
#' @param embedding_dim latent width (default 30).
#' @param max_epochs,lr,batch_size,early_stop_patience,min_delta,val_percentage,weight_batch_ae
#'   Stage-1 training hyperparameters (defaults match the from-scratch driver).
#' @param weight_modality optional per-modality reconstruction-loss weights. A
#'   named list keyed by modality (e.g. \code{list(rna = 1, atac = 0.2)} to
#'   down-weight noisier ATAC), or a numeric vector in rna/adt/atac-present
#'   order; \code{NULL} (default) weights every present modality equally.
#' @param balance_batches train the VAE on a batch-balanced subset (default TRUE).
#' @param donor_prediction force the donor predictor on/off; \code{NULL} (default)
#'   enables it when \code{donor_key} + \code{celltype_key} are present.
#' @param donor_predictor named list of donor-predictor knobs (e.g.
#'   \code{list(glr = 3e-3, n_epochs = 120, adv_max = 10, adv_warmup = 7,
#'   n_disc = 21, patient_w = 10)}); \code{NULL} uses the driver defaults.
#' @param seed integer seed applied before training so the (otherwise unseeded)
#'   Stage-1 VAE is reproducible; \code{NULL} keeps the engine's unseeded
#'   behaviour. Default 0.
#' @param device one of "auto", "cpu", "cuda" (session device, fixed on first use).
#' @return an \code{m3_model}.
#' @examples
#' \donttest{
#'   data  <- m3_demo()
#'   model <- m3_train(data, condition_keys = c("cond_group", "Age_interval"),
#'                     celltype_key = "mergedcelltype", embedding_dim = 30,
#'                     donor_prediction = FALSE, max_epochs = 80)
#'   model
#' }
#' @export
m3_train <- function(data,
                     condition_keys,
                     target_condition = NULL,
                     celltype_key = NULL,
                     donor_key = NULL,
                     batch_key = NULL,
                     held_out = NULL,
                     held_out_samples = NULL,
                     hvg = NULL,
                     embedding_dim = 30L,
                     max_epochs = 300L,
                     lr = 1e-5,
                     batch_size = 256L,
                     early_stop_patience = 300L,
                     min_delta = 0,
                     val_percentage = 0.1,
                     weight_batch_ae = 1,
                     weight_modality = NULL,
                     balance_batches = TRUE,
                     donor_prediction = NULL,
                     donor_predictor = NULL,
                     seed = 0L,
                     device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  if (!inherits(data, "m3_dataset")) {
    data <- m3_dataset(data, device = device)
  }
  meta <- .m3_call(
    "model_train",
    dataset_handle = data$handle,
    condition_keys = as.list(condition_keys),
    target_condition = target_condition,
    celltype_key = celltype_key,
    donor_key = donor_key,
    batch_key = batch_key,
    held_out = if (is.null(held_out)) NULL else as.list(held_out),
    held_out_samples = if (is.null(held_out_samples)) NULL else as.list(held_out_samples),
    hvg = hvg,
    embedding_dim = embedding_dim,
    max_epochs = max_epochs,
    lr = lr,
    batch_size = batch_size,
    early_stop_patience = early_stop_patience,
    min_delta = min_delta,
    val_percentage = val_percentage,
    weight_batch_ae = weight_batch_ae,
    weight_modality = weight_modality,
    balance_batches = balance_batches,
    donor_prediction = donor_prediction,
    donor_predictor = donor_predictor,
    seed = seed,
    .device = data$device
  )
  new_m3_model(meta, data$device)
}

#' Store a trained model inside a SingleCellExperiment / MultiAssayExperiment.
#'
#' Lets the object carry its model (\code{metadata(x)$m3}) so it pipes into the
#' readout verbs, mirroring the Seurat-style flow.
#'
#' @param object an SCE/MAE.
#' @param model an \code{m3_model}.
#' @return \code{object} with the model attached.
#' @export
m3_attach <- function(object, model) {
  stopifnot(inherits(model, "m3_model"))
  S4Vectors::metadata(object)$m3 <- model
  object
}
