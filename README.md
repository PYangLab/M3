# M3

**M3** is a deep generative framework for **condition-aware integration**,
**patient-level inference**, and **multi-resolution interpretation** of multimodal
single-cell omics data across many biological conditions and samples.

From one shared, condition-aware representation, M3 delivers six downstream tasks:
factorised dimension reduction, condition-aware batch correction, mosaic
integration + imputation, patient-level condition inference, patient/sample
generation, and multi-resolution attribution.

📖 **Documentation & tutorials:** https://pyanglab.github.io/M3/

## System requirements

M3 runs on Linux with Python 3.10–3.12. All dependencies are installed automatically via pip (see `pyproject.toml`), and no non-standard hardware is required (a GPU is optional).

## Install
Installation typically takes a few seconds.

### Python

```bash
pip install m3-sc      # imported as `import m3`
```

PyTorch comes along automatically. If you need a build matched to a specific CUDA
version, install it first and pip will keep it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install m3-sc
```

For the development version: `pip install "git+https://github.com/PYangLab/M3.git"`. 

### R

```r
install.packages("remotes")
remotes::install_github("PYangLab/M3", subdir = "m3-r")
```

The R package provisions its own Python engine through
[basilisk](https://bioconductor.org/packages/basilisk/) on first use — no manual
Python / PyTorch setup, and R and Python produce identical results.

## Quickstart (Python)

```python
import m3

data  = m3.datasets.liu_demo()                                  # built-in demo
model = m3.M3(data, condition_keys=["cond_group"],
              celltype_key="mergedcelltype").train()
emb   = model.embedding(part="bio")                             # integrated embedding
```
More demos, together with their expected outputs, are available in the [tutorials](https://pyanglab.github.io/M3/).

## Using M3 on your own data

M3 takes raw counts. Each batch needs a count matrix (RNA and/or ADT and/or ATAC)
and cell metadata with donor, cell type, condition and batch columns. Load batches
with `m3.read_h5ad` (or `m3.read_h5` / `m3.from_anndata`), combine them with `m3.concat`,
then build and train the model as in the Quickstart. See the
[documentation](https://pyanglab.github.io/M3/) for the full workflow.

## Repository layout

| Path | What |
|---|---|
| `src/m3/` | the `m3-sc` Python package (PyTorch engine vendored under `_engine/`) |
| `m3-r/` | the R package — `library(m3)`, basilisk-wrapped, same engine |
| `website/` | documentation site (MkDocs Material); `mkdocs serve -f website/mkdocs.yml` |
| `.github/workflows/deploy.yml` | builds + publishes the docs to GitHub Pages |
| `tests/` | Python test suite |

## License

See [LICENSE](LICENSE).
