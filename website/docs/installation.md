---
tags:
  - getting-started
---

# Installation

M3 is installed from PyPI as **`m3-sc`** and imported as **`import m3`**. It is
tested on **Python 3.10 – 3.12** on Linux and macOS.

!!! tip "TL;DR"
    ```bash
    conda create -n m3 python=3.11 -y && conda activate m3
    pip install m3-sc          # imported as `import m3`; PyTorch comes along automatically
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

## Step 2 — PyTorch *(usually nothing to do)*

M3's training engine runs on **PyTorch**, which `pip install m3-sc` pulls in
automatically as a dependency of `captum`. Most people can skip straight to
Step 3.

=== "Default (CPU / macOS)"

    Nothing to do — Step 3 brings PyTorch with it. On Apple Silicon, M3 runs on
    the CPU backend.

=== "Linux + CUDA 12.x"

    Only if you need a build matched to a specific CUDA version, install it
    *before* Step 3 — pip will then keep the build you chose:

    ```bash title="terminal"
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

Verify, once Step 3 is done:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Step 3 — Install M3

=== "pip (recommended)"

    ```bash title="terminal"
    pip install m3-sc
    ```

    Imported as `import m3`. PyTorch comes along automatically. For the
    development version: `pip install "git+https://github.com/PYangLab/M3.git"`.

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
| `scanpy` | neighbours / UMAP used by the engine |
| `tqdm` | training progress bars |
| `captum` | integrated-gradients attribution — **brings `torch` with it** |

PyTorch therefore arrives automatically, so a plain `pip install m3-sc` gives you
everything the training engine needs. Install `torch` yourself only if you want a
build matched to a specific CUDA version (Step 2).

---

## Step 4 — Tutorial extras *(nothing to do)*

The [tutorial notebooks](notebooks/py_01_representation_learning.ipynb) plot UMAPs
and ROC curves with `scanpy`, `umap-learn` and `matplotlib`. All three arrive with
`m3-sc`: `scanpy` is a declared dependency, and it requires `umap-learn` and
`matplotlib` itself. There is nothing extra to install.

---

## Step 5 — Verify the install

```bash title="terminal"
python -c "import m3; print(m3.__version__)"
```

Expected output: the installed version number. If a version prints, the API is
reachable.

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
metadata with a **batch** column, a **condition / phenotype** column and a
**cell-type** column, and a **donor / sample** column for the donor-level tasks.
Load batches with
[`m3.read_h5`](api-python.md#m3read_h5), [`m3.read_h5ad`](api-python.md#m3read_h5ad),
[`m3.from_anndata`](api-python.md#m3from_anndata) or
[`m3.read_matrix`](api-python.md#m3read_matrix), and combine them with
[`m3.concat`](api-python.md#m3concat). See the [API reference](api-python.md) for exact signatures,
and [Use your own data](notebooks/py_05_own_data.ipynb) for a worked example.

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

    :   `m3-sc` normally brings PyTorch with it, so this points at a broken or
        partial environment. `pip install --force-reinstall torch` into the same
        environment, and check you are not in a different env than you installed into.

    **Still broken?**

    :   [Open an issue](https://github.com/PYangLab/M3/issues/new) and paste the
        output of `python -c "import m3; print(m3.__version__)"` plus your
        `pip freeze`.

---

Next: the [Quickstart](quickstart.md).
