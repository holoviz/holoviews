from datetime import date, datetime

import pandas as pd

from .boundingregion import *
from .data import *
from .dimension import *
from .element import *
from .layout import *
from .operation import *
from .overlay import *
from .sheetcoords import *
from .spaces import *
from .tree import *
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
Dimension.type_formatters[pd.Timestamp] = "%Y-%m-%d %H:%M:%S"


def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Dimension, Dimensioned, Operation, BoundingBox,
                   SheetCoordinateSystem, AttrTree]
    return any([issubclass(obj, bc) for bc in baseclasses])

_public = list({_k for _k, _v in locals().items() if public(_v)})
__all__ = _public + ["boundingregion", "dimension", "layer", "layout",
                     "ndmapping", "operation", "options", "sheetcoords", "tree", "element"]
