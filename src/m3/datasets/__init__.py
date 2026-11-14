"""Built-in demo datasets.

Public entry point:

  >>> import m3
  >>> data = m3.datasets.liu_demo()    # one-line load, no manual subsampling

Returns an :class:`m3.Dataset` ready to feed straight into :class:`m3.M3`.

Models trained in advance on that data, to load instead of training:

  >>> model = m3.M3.load(m3.datasets.liu_demo_model("patient_prediction"), data)
"""

from m3.datasets._loaders import liu_demo, liu_demo_model

__all__ = ["liu_demo", "liu_demo_model"]
