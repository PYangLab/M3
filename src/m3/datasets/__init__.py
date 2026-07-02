"""Built-in demo datasets.

Public entry point:

  >>> import m3
  >>> data = m3.datasets.liu_demo()    # one-line load, no manual subsampling

Returns an :class:`m3.Dataset` ready to feed straight into :class:`m3.M3`.
"""

from m3.datasets._loaders import liu_demo

__all__ = ["liu_demo"]
