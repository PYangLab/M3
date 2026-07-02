"""Signature tests for M3.augment(batch=...)."""
import inspect
from m3._model import M3


def test_augment_has_batch_param():
    sig = inspect.signature(M3.augment)
    assert "batch" in sig.parameters
    p = sig.parameters["batch"]
    assert p.default is None
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
