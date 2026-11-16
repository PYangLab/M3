"""The R package runs a vendored copy of src/m3; it must stay byte-identical."""
import filecmp
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC, VENDORED = ROOT / "src" / "m3", ROOT / "m3-r" / "inst" / "python" / "m3"


def _files(root):
    return {p.relative_to(root) for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


@pytest.mark.skipif(not VENDORED.is_dir(), reason="m3-r is not in this checkout")
def test_vendored_copy_matches_src():
    src, vendored = _files(SRC), _files(VENDORED)
    assert src == vendored, f"only in src: {sorted(src - vendored)}; only in m3-r: {sorted(vendored - src)}"
    differ = [str(p) for p in sorted(src) if not filecmp.cmp(SRC / p, VENDORED / p, shallow=False)]
    assert not differ, f"m3-r/inst/python/m3 differs from src/m3 in: {differ}; copy src/m3 over it"
