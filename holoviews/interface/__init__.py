from ..core import Dimensioned, AttrTree

try:
    import pandas
    from .pandas import DFrame # pyflakes:ignore (API import)
except:
    pandas = None

try:
    import seaborn
    from .seaborn import *     # pyflakes:ignore (API import)
except:
    seaborn = None

from .collector import *       # pyflakes:ignore (API import)

def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Dimensioned, Collector, AttrTree]
    return any([issubclass(obj, bc) for bc in baseclasses])

__all__ = list(set([_k for _k, _v in locals().items() if public(_v)]))

