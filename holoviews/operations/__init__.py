from ..core.operation import ViewOperation, StackOperation

from channel import * # pyflakes:ignore (API import)
from stack import * # pyflakes:ignore (API import)
from view import * # pyflakes:ignore (API import)


def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [ViewOperation, StackOperation]
    return any([issubclass(obj, bc) for bc in baseclasses])

_public = list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public