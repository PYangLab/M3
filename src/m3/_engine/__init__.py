"""Not part of the public API. The user-facing :class:`m3.M3` orchestrates these
modules. Modules keep their original relative imports (``from .util import ...``)
and run unchanged; only their location moved into the ``m3._engine`` package.
"""
