#!/usr/bin/env python3
"""Generate runnable Google Colab notebooks for the 4 Python + 4 R tutorials.

The notebooks / pages shown on the docs site are the source of truth and are NOT
modified. This derives Colab-ready copies into website/colab/, each with a setup
cell prepended so "Open in Colab -> Run all" works on a fresh machine:

  - Python: docs/notebooks/py_0N_*.ipynb  + %pip install cell   (outputs cleared)
  - R:      docs/notebooks/r_0N_*.md       -> R-kernel notebook + install cell
            (the knitr `#>` output lines and the media <img> tags are stripped,
             since Colab regenerates them on run)

The "Open in Colab" buttons (overrides/main.html) link to these files on GitHub.
Re-run after editing any tutorial:

    python website/tools/gen_colab_notebooks.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

WEBSITE = Path(__file__).resolve().parent.parent      # website/
NB = WEBSITE / "docs" / "notebooks"
OUT = WEBSITE / "colab"

PY_TUTORIALS = ["py_01_representation_learning", "py_02_patient_prediction",
                "py_03_attribution", "py_04_augmentation"]
R_TUTORIALS = ["r_01_representation_learning", "r_02_patient_prediction",
               "r_03_attribution", "r_04_augmentation"]

# Why the install cell is a few lines of Python and not `%pip install m3-sc`:
#
# Recent scverse releases float ahead of Colab's preinstalled stack -- scanpy 1.12.1
# wants pandas>=2.3 and anndata 0.13 wants numpy>=2.1, while Colab currently ships
# numpy 2.0.2 / pandas 2.2.2. An unconstrained install therefore upgrades numpy and
# pandas, and replacing those compiled packages under a live kernel leaves a
# half-loaded numpy: the very next `import scanpy` dies with "cannot import name
# '_center' from numpy._core.umath". Run all used to break at the second cell and
# only a kernel restart cleared it.
#
# The cell pins numpy and pandas to whatever the runtime already has, and lets pip
# work out which scanpy/anndata fit. Two reasons to do it this way round rather than
# capping scanpy/anndata ourselves:
#   - no version literals to go stale. When Colab updates its stack, the same cell
#     starts resolving to newer scverse on its own, with no edit here.
#   - pip is better at it. A hand-written `scanpy<1.12` looked right but was wrong:
#     scanpy 1.12.0 requires pandas>=2.2.2, exactly Colab's version, and only 1.12.1
#     raised it to >=2.3. The cap would have held users on 1.11.5 for no reason.
#
# Verified against real pip on both baselines. numpy 2.0.2 / pandas 2.2.2 resolves
# to scanpy 1.12 / anndata 0.12.19; numpy 2.4.6 / pandas 3.0.5 resolves to scanpy
# 1.12.3 / anndata 0.13.2. Neither moves numpy or pandas, so neither needs a
# restart, and the 38-test suite passes on the older resolution too.
#
# Why the pynndescent step follows:
#
# Holding pandas still has a cost. Colab's pandas (2.2.3 as of 2026-11) keeps scanpy
# at 1.12, since every release from 1.12.1 on requires pandas>=2.3 -- but Colab also
# preinstalls pynndescent 0.7.0, whose transformer returns a scipy sparse *array*.
# scanpy before 1.12.5 calls `.getnnz()` on that result, a method sparse arrays do
# not have, so `sc.pp.neighbors` raised "'csr_array' object has no attribute
# 'getnnz'" in every tutorial's UMAP cell. Only the pynndescent path is affected,
# i.e. more than 4096 cells; smaller inputs go through sklearn and work.
#
# pip cannot see this: scanpy's metadata asks only for pynndescent>=0.5.13. So the
# cell checks which scanpy pip actually resolved and, only when it predates the fix,
# moves pynndescent back below 0.7. The guard expires by itself -- once Colab ships
# pandas>=2.3 the first install resolves to scanpy>=1.12.5 and pynndescent is left
# alone. Both literals are release history, not a guess about the future: 0.7 is
# where pynndescent changed its return type, 1.12.5 is where scanpy stopped assuming
# the old one. pynndescent is pure Python and not yet imported when this cell runs,
# so replacing it needs no restart; the numpy/pandas pins ride along on the second
# install so it cannot move them either.
#
# Verified by running, on Python 3.13 / numpy 2.4.6 / pandas 2.2.3 / scipy 1.16.3:
# scanpy 1.12 + pynndescent 0.7.0 reproduces the error on 30k cells; scanpy 1.12 +
# pynndescent 0.6.0 passes; scanpy 1.12.5 + pynndescent 0.7.0 (pandas 3.1.1) passes.
#
# The published package carries no such constraint: `pip install m3-sc` resolves to
# the current stack, which is what CI tests. umap-learn and matplotlib are not
# listed here because scanpy requires both itself.
#
# The generated cell is deliberately comment-free -- the reasoning lives here rather
# than in the reader's first screen. Keep the numpy/pandas arguments though: without
# them the install upgrades both and the tutorial breaks at the next cell.
PY_INSTALL = '''import subprocess, sys, numpy, pandas
from importlib.metadata import version
from packaging.version import Version

pins = [f"numpy=={numpy.__version__}", f"pandas=={pandas.__version__}"]

def install(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages, *pins],
                   check=True)

install("m3-sc")
if Version(version("scanpy")) < Version("1.12.5"):
    install("pynndescent<0.7")'''

R_INSTALL = '''# Colab setup - run this first, then switch the runtime to R
# (Runtime > Change runtime type > R). Precompiled Linux binaries via Posit P3M
# keep the install to a couple of minutes instead of compiling from source.
local({
  cn <- tryCatch(system("lsb_release -cs", intern = TRUE), error = function(e) "jammy")
  options(repos = c(CRAN = sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", cn)),
          HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
            paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))
})
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(
  c("SingleCellExperiment", "SummarizedExperiment", "MultiAssayExperiment",
    "S4Vectors", "basilisk"),
  update = FALSE, ask = FALSE)
install.packages(c("remotes", "reticulate", "Matrix", "ggplot2"))
remotes::install_github("PYangLab/M3", subdir = "m3-r")
# The first m3_train() builds the bundled Python engine via basilisk (a few
# minutes on first run); afterwards it is cached for the session.'''

FENCE = re.compile(r"```[ ]*(\w*)\n(.*?)```", re.DOTALL)   # ``` r  /  ``` blocks
IMG_TAG = re.compile(r"<img[^>]*>")                        # raw media <img ...>
OUT_LINE = re.compile(r"^#>.*$", re.MULTILINE)             # knitr output lines


def _lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(src)}


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def _finalize(cells: list[dict], metadata: dict) -> dict:
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def gen_python(name: str) -> int:
    nb = json.loads((NB / f"{name}.ipynb").read_text(encoding="utf-8"))
    cells = [_code(PY_INSTALL)]
    for c in nb["cells"]:
        if c.get("cell_type") == "code":          # clear outputs -> lean, run-fresh
            c = {**c, "outputs": [], "execution_count": None}
        cells.append(c)
    out = _finalize(cells, nb.get("metadata", {}))
    (OUT / f"{name}.ipynb").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(cells)


def gen_r(name: str) -> int:
    text = IMG_TAG.sub("", (NB / f"{name}.md").read_text(encoding="utf-8"))
    cells: list[dict] = [_code(R_INSTALL)]
    pos = 0
    for m in FENCE.finditer(text):
        prose = text[pos:m.start()].strip("\n")
        if prose.strip():
            cells.append(_md(prose))
        lang, code = m.group(1), m.group(2)
        if lang == "r":
            code = OUT_LINE.sub("", code)                        # drop `#>` output
            code = re.sub(r"\n{3,}", "\n\n", code).strip("\n")   # tidy blank runs
            cells.append(_code(code))
        else:
            cells.append(_md(m.group(0)))
        pos = m.end()
    tail = text[pos:].strip("\n")
    if tail.strip():
        cells.append(_md(tail))
    # Tutorials plot with ggplot2, but the source Rmd attaches it in a hidden
    # setup chunk; inject library(ggplot2) next to library(m3) so "Run all" works.
    for c in cells:
        if c["cell_type"] == "code":
            idx = next((i for i, ln in enumerate(c["source"]) if "library(m3)" in ln), None)
            if idx is not None:
                c["source"].insert(idx + 1, "library(ggplot2)" + chr(10))
                break
    meta = {
        "kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
        "language_info": {"name": "R"},
        "colab": {"provenance": []},
    }
    out = _finalize(cells, meta)
    (OUT / f"{name}.ipynb").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(cells)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for n in PY_TUTORIALS:
        print(f"  colab/{n}.ipynb : {gen_python(n)} cells")
    for n in R_TUTORIALS:
        print(f"  colab/{n}.ipynb : {gen_r(n)} cells")
