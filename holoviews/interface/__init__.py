from ..core import Dimensioned, AttrTree

from .collector import *       # noqa (API import)

def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Dimensioned, Collector, AttrTree]
    return any([issubclass(obj, bc) for bc in baseclasses])

__all__ = list(set([_k for _k, _v in locals().items() if public(_v)]))

