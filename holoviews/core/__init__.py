from datetime import date, datetime

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
from .util import config       # noqa (API import)
from .io import FileArchive

archive = FileArchive()

# Define default type formatters
Dimension.type_formatters[int] = "%d"
Dimension.type_formatters[np.uint16] = '%d'
Dimension.type_formatters[np.int16] = '%d'
Dimension.type_formatters[np.uint32] = '%d'
Dimension.type_formatters[np.int32] = '%d'
Dimension.type_formatters[np.uint64] = '%d'
Dimension.type_formatters[np.int64] = '%d'
Dimension.type_formatters[float] = "%.5g"
Dimension.type_formatters[np.float32] = "%.5g"
Dimension.type_formatters[np.float64] = "%.5g"
Dimension.type_formatters[np.datetime64] = '%Y-%m-%d %H:%M:%S'
Dimension.type_formatters[datetime] = '%Y-%m-%d %H:%M:%S'
Dimension.type_formatters[date] = '%Y-%m-%d'

try:
    import pandas as pd
    Dimension.type_formatters[pd.Timestamp] = "%Y-%m-%d %H:%M:%S"
except:
    pass

def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Dimension, Dimensioned, Operation, BoundingBox,
                   SheetCoordinateSystem, AttrTree]
    return any([issubclass(obj, bc) for bc in baseclasses])

_public = list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public + ["boundingregion", "dimension", "layer", "layout",
                     "ndmapping", "operation", "options", "sheetcoords", "tree", "element"]
