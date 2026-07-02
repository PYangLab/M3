---
tags:
  - getting-started
---

# Installation

M3 is installed from GitHub (a PyPI release as **`m3-sc`** is planned) and imported as **`import m3`**. It is
tested on **Python 3.10 – 3.12** on Linux and macOS.

!!! tip "TL;DR"
    ```bash
    conda create -n m3 python=3.11 -y && conda activate m3
    pip install torch          # match your CUDA setup first
    pip install "git+https://github.com/PYangLab/M3.git"   # imported as `import m3`
    ```

---

## Step 1 — Create an isolated environment

=== "Conda (recommended)"

    ```bash title="terminal"
    conda create -n m3 python=3.11 -y
    conda activate m3
    ```

=== "Mamba (faster solver)"

    ```bash title="terminal"
    mamba create -n m3 python=3.11 -y
    mamba activate m3
    ```

=== "venv"

    ```bash title="terminal"
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```

---

## Step 2 — Install PyTorch

M3's training engine runs on **PyTorch**, which is not pulled in automatically so
that you can match it to your CUDA / MPS setup. Install it *first*.

=== "CPU"

    ```bash title="terminal"
    pip install torch
    ```

=== "Linux + CUDA 12.x"

    ```bash title="terminal"
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

=== "macOS (Apple Silicon)"

    ```bash title="terminal"
    pip install torch    # M3 runs on the CPU backend on Apple Silicon
    ```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Step 3 — Install M3

=== "pip (recommended)"

    ```bash title="terminal"
    pip install "git+https://github.com/PYangLab/M3.git"
    ```

    Imported as `import m3`. A PyPI release as `m3-sc` is planned.

=== "From source"

    ```bash title="terminal"
    git clone https://github.com/PYangLab/M3.git
    cd M3
    pip install -e .
    ```

Installing `m3-sc` pulls in its runtime dependencies:

| Package | Used for |
|---|---|
| `numpy` | array math |
| `scipy` | sparse count matrices |
| `pandas` | metadata I/O |
| `anndata` | `.h5ad` reading |
| `h5py` | paper-format expression matrix I/O |
| `scikit-learn` | internal PCA / scaling in the engine |

PyTorch (Step 2) is required at training time but installed separately.

---

## Step 4 — Tutorial extras *(optional)*

The [tutorial notebooks](notebooks/py_01_representation_learning.ipynb) plot UMAPs
and ROC curves, which need a few extra libraries that the core package does not
require:

```bash title="terminal"
pip install scanpy umap-learn matplotlib
```

---

## Step 5 — Verify the install

```bash title="terminal"
python -c "import m3; print(m3.__version__)"
```

Expected output: the installed version string (e.g. `0.3.0.dev0`). If a version
prints, the API is reachable.

The demo dataset used throughout the tutorials is **built into the wheel** — no
download or preprocessing step:

```python
import m3
data = m3.datasets.liu_demo()
print(data)
# Dataset(n_cells=30534, batches=['B1', 'B2', 'B3'], modalities=[rna:1000, adt:192])
```

You're ready for the [Quickstart](quickstart.md).

---

## Installing the R package

The R interface lives in the **same repository** (the `m3-r/` directory) and wraps
the identical engine, so R and Python produce the same results. It installs
straight from GitHub — no Bioconductor/CRAN step:

```r
install.packages("remotes")
remotes::install_github("PYangLab/M3", subdir = "m3-r")
```

You do **not** install Python or PyTorch yourself — on first use the R package
provisions its own engine environment through
[basilisk](https://bioconductor.org/packages/basilisk/). Verify:

```r
library(m3)
data <- m3_demo()
data
```

See the [R tutorials](notebooks/r_01_representation_learning.md) and the
[R API](api-r.md).

---

## GPU notes

- If a CUDA-enabled `torch` is detected, M3 uses it automatically — no `.to("cuda")`
  calls anywhere in the user-facing API.
- For the built-in demo, CPU is fine. For full real-data scale (≥ 100 k cells,
  `max_epochs=500`), a GPU saves roughly 10× wall time.
- On Apple Silicon, M3 runs on the **CPU** backend — the engine selects CUDA
  when a GPU is present and otherwise CPU (MPS acceleration is not yet enabled).

---

## Input data format

M3 takes **raw counts** — it normalises internally, so do **not** pre–log-normalise.
Each batch is a multimodal table (RNA and/or ADT and/or ATAC) plus per-cell
metadata with at least a **donor / sample** column, a **cell-type** column, a
**condition / phenotype** column, and a **batch** column. Load batches with
[`m3.read_h5`](api-python.md#m3read_h5), [`m3.read_h5ad`](api-python.md#m3read_h5ad), or
[`m3.from_anndata`](api-python.md#m3from_anndata), and combine them with
[`m3.concat`](api-python.md#m3concat). See the [API reference](api-python.md) for exact signatures.

---

## Troubleshooting

!!! question "Stuck on install?"

    **GPU / CUDA mismatch error** at `import torch`

    :   The pre-built `torch` wheel does not match your CUDA toolkit.
        Reinstall via the [PyTorch selector](https://pytorch.org/get-started/locally/)
        with the correct `--index-url`.

    **`conda` solver hangs for > 2 minutes**

    :   Switch to mamba: `conda install -n base -c conda-forge mamba`, then
        re-create the env with `mamba`. Or `conda config --set solver libmamba`
        to use libmamba globally.

    **`ModuleNotFoundError: No module named 'torch'` when you `import m3`**

    :   PyTorch is a separate install (Step 2). `pip install torch` into the same
        environment.

    **Still broken?**

    :   [Open an issue](https://github.com/PYangLab/M3/issues/new) and paste the
        output of `python -c "import m3; print(m3.__version__)"` plus your
        `pip freeze`.

---

Next: the [Quickstart](quickstart.md).
