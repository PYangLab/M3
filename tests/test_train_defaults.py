"""Quick assertion that train() defaults match the documented contract."""
import inspect
from m3._model import M3


def test_train_balance_batches_defaults_true():
    sig = inspect.signature(M3.train)
    assert sig.parameters["balance_batches"].default is True
