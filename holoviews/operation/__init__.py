from ..core.operation import Operation                # noqa (API import)
from ..core.options import Compositor

from .element import *      # noqa (API import)
from ..core import Overlay  # noqa (API import)

def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Operation]
    return any([issubclass(obj, bc) for bc in baseclasses])


_public = list(set([_k for _k, _v in locals().items() if public(_v)]))

_current_locals = [el for el in locals().items()]
for _k, _v in _current_locals:
    if public(_v) and issubclass(_v, Operation):
        Compositor.operations.append(_v)

__all__ = _public + ['Compositor']
