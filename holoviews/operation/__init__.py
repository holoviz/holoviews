from ..core.operation import ElementOperation, MapOperation
from ..core.options import Channel

from .rgb import *          # pyflakes:ignore (API import)
from .element import *      # pyflakes:ignore (API import)
from ..core import Overlay  # pyflakes:ignore (API import)
from .map import *          # pyflakes:ignore (API import)


def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [ElementOperation, MapOperation]
    return any([issubclass(obj, bc) for bc in baseclasses])


_public = list(set([_k for _k, _v in locals().items() if public(_v)]))

for _k, _v in locals().items():
    if public(_v) and issubclass(_v, ElementOperation):
        Channel.operations.append(_v)

__all__ = _public + ['Channel']
