from ..core.operation import ViewOperation, MapOperation

from .channel import * # pyflakes:ignore (API import)
from .map import * # pyflakes:ignore (API import)
from .view import * # pyflakes:ignore (API import)


def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [ViewOperation, MapOperation]
    return any([issubclass(obj, bc) for bc in baseclasses])

_public = list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public