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
