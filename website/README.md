# M3 documentation site

Source for the M3 documentation site, built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) +
[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter). Published to
<https://pyanglab.github.io/M3/> by the GitHub Actions workflow in
`.github/workflows/deploy.yml` on every push to `main`.

## Build / preview locally

```bash
pip install -r requirements-docs.txt
mkdocs serve -f mkdocs.yml            # http://127.0.0.1:8000
mkdocs build --strict -f mkdocs.yml   # into ./site
```

## Layout

```
website/
  mkdocs.yml                # nav, theme, plugins, markdown extensions
  requirements-docs.txt     # pinned docs toolchain (used by CI)
  overrides/                # theme overrides (custom 404, Colab button)
  docs/
    index.md                # landing page
    installation.md         # install for Python + R
    quickstart.md           # end-to-end on the built-in demo
    overview.md             # "How M3 works"
    api-python.md           # Python API reference
    api-r.md                # R API reference
    faq.md                  # frequently asked questions
    notebooks/              # rendered Python (.ipynb) + R (.md) tutorials
    assets/                 # logo + architecture figure
    stylesheets/ · javascripts/ · includes/
  colab/                    # runnable Colab copies of the tutorials
  tools/
    gen_colab_notebooks.py  # regenerate the colab/ copies from docs/notebooks/
    trim_epoch_logs.py      # trim per-epoch loss noise after re-rendering
```
