from .boundingregion import *  # noqa (API import)
from .data import *            # noqa (API import)
from .dimension import *       # noqa (API import)
from .element import *         # noqa (API import)
from .layout import *          # noqa (API import)
from .operation import *       # noqa (API import)
from .overlay import *         # noqa (API import)
from .sheetcoords import *     # noqa (API import)
from .spaces import *          # noqa (API import)
from .tree import *            # noqa (API import)
from .io import FileArchive

archive = FileArchive()

# Define default type formatters
Dimension.type_formatters[int] = "%d"
Dimension.type_formatters[float] = "%.5g"
Dimension.type_formatters[np.float32] = "%.5g"
Dimension.type_formatters[np.float64] = "%.5g"


def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Dimension, Dimensioned, ElementOperation, BoundingBox,
                   SheetCoordinateSystem, AttrTree]
    return any([issubclass(obj, bc) for bc in baseclasses])

_public = list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public + ["boundingregion", "dimension", "layer", "layout",
                     "ndmapping", "operation", "options", "sheetcoords", "tree", "element"]

