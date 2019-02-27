import numpy as np

import param

from ..core import Dimension, Dataset, Element2D
from ..streams import BoundsXY


class Geometry(Dataset, Element2D):
    """
    Geometry elements represent a collection of objects drawn in
    a 2D coordinate system. The two key dimensions correspond to the
    x- and y-coordinates in the 2D space, while the value dimensions
    may be used to control other visual attributes of the Geometry
    """

    group = param.String(default='Geometry', constant=True)

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True, doc="""
        The key dimensions of a geometry represent the x- and y-
        coordinates in a 2D space.""")

    vdims = param.List(default=[], constant=True, doc="""
        Value dimensions can be associated with a geometry.""")

    __abstract = True


class GeometrySelectionExpr(object):
    """
    Mixin class for Geometry elements to add basic support for
    SelectionExpr streams.
    """
    _selection_streams = (BoundsXY,)

    def _get_selection_expr_for_stream_value(self, **kwargs):
        from ..util.transform import dim

        invert_axes = self.opts.get('plot').kwargs.get('invert_axes', False)

        if kwargs.get('bounds', None):
            x0, y0, x1, y1 = kwargs['bounds']

            # Handle invert_xaxis/invert_yaxis
            if y0 > y1:
                y0, y1 = y1, y0
            if x0 > x1:
                x0, x1 = x1, x0

            if invert_axes:
                ydim, xdim = self.kdims[:2]
            else:
                xdim, ydim = self.kdims[:2]

            bbox = {
                xdim.name: (x0, x1),
                ydim.name: (y0, y1),
            }

            selection_expr = (
                    (dim(xdim) >= x0) & (dim(xdim) <= x1) &
                    (dim(ydim) >= y0) & (dim(ydim) <= y1)
            )

            return selection_expr, bbox
        return None, None


class Points(GeometrySelectionExpr, Geometry):
    """
    Points represents a set of coordinates in 2D space, which may
    optionally be associated with any number of value dimensions.
    """

    group = param.String(default='Points', constant=True)

    _auto_indexable_1d = True


class VectorField(GeometrySelectionExpr, Geometry):
    """
    A VectorField represents a set of vectors in 2D space with an
    associated angle, as well as an optional magnitude and any number
    of other value dimensions. The angles are assumed to be defined in
    radians and by default the magnitude is assumed to be normalized
    to be between 0 and 1.
    """

    group = param.String(default='VectorField', constant=True)

    vdims = param.List(default=[Dimension('Angle', cyclic=True, range=(0,2*np.pi)),
                                Dimension('Magnitude')], bounds=(1, None))


class Segments(Geometry):
    """
    Segments represent a collection of lines in 2D space.
    """
    group = param.String(default='Segments', constant=True)

    kdims = param.List(default=[Dimension('x0'), Dimension('y0'),
                                Dimension('x1'), Dimension('y1')],
                       bounds=(4, None), constant=True, doc="""
        Segments represent lines given by x- and y-
        coordinates in 2D space.""")
