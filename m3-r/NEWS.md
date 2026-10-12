# m3 (development version)

* `m3_train(seed=)` now reaches training. It was overridden by the engine's own
  default, so every run used seed 0 whatever was passed. **Results change for any
  non-zero seed**; `seed = NULL` now gives the unseeded behaviour it documents.
* `batch_key` now selects the batches that are integrated. Integration always
  grouped by the column literally named `batch`, so a different `batch_key` was
  silently trained as one batch with no batch correction. **Results change when
  `batch_key` is not `"batch"`.** A single-batch dataset now warns.
* Inputs keep their cell names in an `obs_names` column, returned by
  `m3_cell_metadata()`. Read-out rows are not guaranteed to be in input order, so
  align on that column rather than by position.

# m3 0.3.0

* First R release: object-in / object-out interface to the `m3-sc` Python engine.
* `m3_train()` trains the integration VAE (+ optional donor predictor) on a
  `SingleCellExperiment` / `MultiAssayExperiment` / matrices, returning an
  `m3_model`.
* Readout verbs: `m3_embedding()`, `m3_reconstruct()`, `m3_predict_donors()`,
  `m3_donor_embedding()`, `m3_attribute()` (+ `m3_top_genes()` /
  `m3_top_celltypes()`), `m3_generate()`, `m3_augment()`.
* Input helpers: `m3_read_h5()`, `m3_concat()`, `m3_dataset()`, `m3_demo()`.
* The Python engine is vendored under `inst/python/m3` and run via `basilisk`.
* `m3_train(seed=)` seeds the Stage-1 VAE (which the upstream engine leaves
  unseeded), making runs reproducible and identical to the Python package.
