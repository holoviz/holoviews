from ..core import Overlay  # noqa (API import)
from ..core.operation import Operation
from ..core.options import Compositor
from .element import (  # noqa: F401
    apply_when,
    categorical_aggregate2d,
    chain,
    collapse,
    contours,
    convolve,
    decimate,
    dendrogram,
    factory,
    function,
    gradient,
    gridmatrix,
    histogram,
    image_overlay,
    interpolate_curve,
    method,
    operation,
    threshold,
    transform,
)


for _name in dir():
    _obj = globals()[_name]
    if isinstance(_obj, type) and issubclass(_obj, Operation) and _obj is not Operation:
        Compositor.operations.append(_obj)


__all__ = [
    "Compositor",
    "Operation",
    "categorical_aggregate2d",
    "chain",
    "collapse",
    "contours",
    "convolve",
    "decimate",
    "dendrogram",
    "factory",
    "function",
    "gradient",
    "histogram",
    "image_overlay",
    "interpolate_curve",
    "method",
    "operation",
    "threshold",
    "transform",
]
