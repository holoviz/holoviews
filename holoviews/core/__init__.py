from .boundingregion import *  # pyflakes:ignore (API import)
from .dimension import *       # pyflakes:ignore (API import)
from .element import *         # pyflakes:ignore (API import)
from .layout import *          # pyflakes:ignore (API import)
from .operation import *       # pyflakes:ignore (API import)
from .overlay import *         # pyflakes:ignore (API import)
from .sheetcoords import *     # pyflakes:ignore (API import)
from .tree import *            # pyflakes:ignore (API import)
from .io import FileArchive

archive = FileArchive()


def formatter(fmt):
    def inner(x): return (fmt % x)
    return inner

# Define default type formatters
Dimension.type_formatters[int] = formatter("%d")
Dimension.type_formatters[float] = formatter("%.3g")
Dimension.type_formatters[np.float32] = formatter("%.3g")
Dimension.type_formatters[np.float64] = formatter("%.3g")


def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Dimension, Dimensioned, ElementOperation, BoundingBox,
                   SheetCoordinateSystem, AttrTree]
    return any([issubclass(obj, bc) for bc in baseclasses])

_public = list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public + ["boundingregion", "dimension", "layer", "layout",
                     "ndmapping", "operation", "options", "sheetcoords", "tree", "element"]

