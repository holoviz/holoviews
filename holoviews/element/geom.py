import numpy as np

import param

from ..core import Dimension, Dataset, Element2D
from ..core.util import lzip, unique_zip
from ..streams import SelectionXY


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


class Selection2DExpr(object):
    """
    Mixin class for Cartesian 2D elements to add basic support for
    SelectionExpr streams.
    """

    _selection_dims = 2

    _selection_streams = (SelectionXY,)

    def _get_selection_expr_for_stream_value(self, **kwargs):
        from ..util.transform import dim
        from .graphs import Graph

        if kwargs.get('bounds') is None and kwargs.get('x_selection') is None:
            return None, None, Rectangles([])

        invert_axes = self.opts.get('plot').kwargs.get('invert_axes', False)

        xcats, ycats = None, None
        x0, y0, x1, y1 = kwargs['bounds']
        if 'x_selection' in kwargs:
            xsel = kwargs['x_selection']
            if isinstance(xsel, list):
                xcats = xsel
                x0, x1 = int(round(x0)), int(round(x1))
            ysel = kwargs['y_selection']
            if isinstance(ysel, list):
                ycats = ysel
                y0, y1 = int(round(y0)), int(round(y1))
        xsel, ysel = (x0, x1), (y0, y1)

        # Handle invert_xaxis/invert_yaxis
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        bounds = (x0, y0, x1, y1)

        if isinstance(self, Graph):
            xdim, ydim = self.nodes.dimensions()[:2]
        else:
            xdim, ydim = self.dimensions()[:2]
        if invert_axes:
            xdim, ydim = ydim, xdim

        bbox = {xdim.name: xsel, ydim.name: ysel}
        index_cols = kwargs.get('index_cols')
        if index_cols:
            index_cols = [self.get_dimension(c) for c in index_cols]
            sel = self.dataset.select(**bbox)
            other = dim(c for c in index_cols[1:])
            vals = dim(index_cols[0], unique_zip, *other).apply(sel)
            selection_expr = dim(index_cols[0], lzip, *other).isin(vals)
            region_element = None
        else:
            if xcats:
                xexpr = dim(xdim).isin(xcats)
            else:
                xexpr = (dim(xdim) >= x0) & (dim(xdim) <= x1)
            if ycats:
                yexpr = dim(ydim).isin(ycats)
            else:
                yexpr = (dim(ydim) >= y0) & (dim(ydim) <= y1)
            selection_expr = (xexpr & yexpr)
            region_element = Rectangles([bounds])
        return selection_expr, bbox, region_element

    @staticmethod
    def _merge_regions(region1, region2, operation):
        if region1 is None or operation == "overwrite":
            return region2
        return region1.clone(region1.interface.concatenate([region1, region2]))



class Points(Selection2DExpr, Geometry):
    """
    Points represents a set of coordinates in 2D space, which may
    optionally be associated with any number of value dimensions.
    """

    group = param.String(default='Points', constant=True)

    _auto_indexable_1d = True


class VectorField(Selection2DExpr, Geometry):
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
                       bounds=(4, 4), constant=True, doc="""
        Segments represent lines given by x- and y-
        coordinates in 2D space.""")


class Rectangles(Geometry):
    """
    Rectangles represent a collection of axis-aligned rectangles in 2D space.
    """

    group = param.String(default='Rectangles', constant=True)

    kdims = param.List(default=[Dimension('x0'), Dimension('y0'),
                                Dimension('x1'), Dimension('y1')],
                       bounds=(4, 4), constant=True, doc="""
        The key dimensions of the Rectangles element represent the
        bottom-left (x0, y0) and top right (x1, y1) coordinates
        of each box.""")

