#!/usr/bin/env python3
"""Train the models that tutorials 2 and 3 can load instead of training.

Each model is made by running the tutorial's own code cells, from the top through
its training cell, and then `model.save(...)`. The shipped file is therefore the
model that page trains, with no second copy of the settings to keep in step.

The files go into the package data (`src/m3/datasets/_data/`, read by
`m3.datasets.liu_demo_model`) and into the copy of the engine vendored in the R
package. Training takes about 40 minutes per model on a laptop CPU.

Run it, then re-execute the two tutorials so the pages show these models, when
    - a tutorial's training settings change, or
    - a change to m3 alters what training produces, or
    - the demo data changes (the shipped models then stop loading: the test
      suite fails in tests/test_demo_models.py).

    python website/tools/train_demo_models.py

Run it on the machine that executes the tutorial pages. Training is repeatable
on one machine but not across machines: a CPU and a GPU, or two library
versions, give different weights, and the donor predictions and attribution
scores move with them (on the full demo, held-out accuracy 1.000 on one machine
and 0.957 on another). A model made elsewhere therefore disagrees with the
numbers the pages show until the tutorials are re-executed with it. The training
cell's log on each page tells the two apart: its first lines are those of the
run that made the shipped model.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "website" / "docs" / "notebooks"
PACKAGE_DATA = REPO / "src" / "m3" / "datasets" / "_data"
VENDORED_DATA = REPO / "m3-r" / "inst" / "python" / "m3" / "datasets" / "_data"

# shipped model name -> the tutorial that trains it
TUTORIALS = {
    "patient_prediction": "py_02_patient_prediction",
    "attribution": "py_03_attribution",
}


def train_as_in(notebook: Path):
    """Run the notebook's code cells through its training cell; return the model."""
    cells = ["".join(c["source"]) for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    last = next((i for i, src in enumerate(cells) if ".train(" in src), None)
    if last is None:
        raise ValueError(f"{notebook.name} has no training cell.")
    namespace: dict = {}
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:   # the tutorials write their outputs to cwd
        os.chdir(scratch)
        try:
            for src in cells[: last + 1]:
                exec(compile(src, notebook.name, "exec"), namespace)
        finally:
            os.chdir(here)
    return namespace["model"]


def build(name: str, notebook: Path, out_dir: Path) -> Path:
    model = train_as_in(notebook)
    path = out_dir / f"liu_demo_{name}.m3"
    model.save(path)
    return path


if __name__ == "__main__":
    for name, tutorial in TUTORIALS.items():
        path = build(name, NOTEBOOKS / f"{tutorial}.ipynb", PACKAGE_DATA)
        shutil.copyfile(path, VENDORED_DATA / path.name)
        print(f"{path.relative_to(REPO)}: {path.stat().st_size / 1e6:.2f} MB")
