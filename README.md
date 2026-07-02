# M3

**M3** is a deep generative framework for **condition-aware integration**,
**patient-level inference**, and **multi-resolution interpretation** of multimodal
single-cell omics data across many biological conditions and samples.

From one shared, condition-aware representation, M3 delivers six downstream tasks:
factorised dimension reduction, condition-aware batch correction, mosaic
integration + imputation, patient-level condition inference, patient/sample
generation, and multi-resolution attribution.

📖 **Documentation & tutorials:** https://pyanglab.github.io/M3/

## Install

### Python

```bash
pip install torch      # install first, matched to your CUDA / CPU setup
pip install "git+https://github.com/PYangLab/M3.git"   # imported as `import m3`
```

_A PyPI release (`pip install m3-sc`) is planned._

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
