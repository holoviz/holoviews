from datetime import date, datetime

import numpy as np

from .accessors import Apply, Redim  # noqa (API import)
from .boundingregion import AARectangle, BoundingBox, BoundingEllipse  # noqa: F401
from .data import Dataset
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .element import Collator, Element, Element2D, Element3D, Tabular
from .io import FileArchive
from .layout import AdjointLayout, Empty, Layout, NdLayout
from .ndmapping import MultiDimensionalMapping, NdMapping, UniformNdMapping
from .operation import Operation
from .options import Store, StoreOptions  # noqa (API import)
from .overlay import CompositeOverlay, NdOverlay, Overlay
from .sheetcoords import SheetCoordinateSystem
from .spaces import DynamicMap, GridMatrix, GridSpace, HoloMap
from .tree import AttrTree
from .util import config  # noqa (API import)

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
Dimension.type_formatters['pandas._libs.tslibs.timestamps.Timestamp'] = "%Y-%m-%d %H:%M:%S"


__all__ = [
    "AdjointLayout",
    "AttrTree",
    "BoundingBox",
    "BoundingEllipse",
    "Collator",
    "CompositeOverlay",
    "Dataset",
    "Dimension",
    "Dimensioned",
    "DynamicMap",
    "Element",
    "Element2D",
    "Element3D",
    "Empty",
    "GridMatrix",
    "GridSpace",
    "HoloMap",
    "Layout",
    "MultiDimensionalMapping",
    "NdLayout",
    "NdMapping",
    "NdOverlay",
    "Operation",
    "Overlay",
    "SheetCoordinateSystem",
    "Tabular",
    "UniformNdMapping",
    "ViewableElement",
    "ViewableTree",
    "boundingregion",
    "dimension",
    "element",
    "layout",
    "ndmapping",
    "operation",
    "options",
    "sheetcoords",
    "tree",
]
