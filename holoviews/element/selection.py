"""
Defines mix-in classes to handle support for linked brushing on
elements.
"""

from ..core import util, NdOverlay
from ..streams import SelectionXY, Selection1D
from ..util.transform import dim
from .annotation import HSpan, VSpan


class SelectionIndexExpr(object):

    _selection_dims = None

    _selection_streams = (Selection1D,)

    def _get_selection_expr_for_stream_value(self, **kwargs):
        index = kwargs.get('index')
        index_cols = kwargs.get('index_cols')
        if index is None or index_cols is None:
            expr = None
        else:
            index_cols = [self.get_dimension(c) for c in index_cols]
            vals = dim(index_cols[0], util.unique_zip, *index_cols[1:]).apply(self.iloc[index])
            expr = dim(index_cols[0], util.lzip, *index_cols[1:]).isin(vals)
        return expr, None, None

    @staticmethod
    def _merge_regions(region1, region2, operation):
        return None


class Selection2DExpr(object):
    """
    Mixin class for Cartesian 2D elements to add basic support for
    SelectionExpr streams.
    """

    _selection_dims = 2

    _selection_streams = (SelectionXY,)

    def _get_selection_expr_for_stream_value(self, **kwargs):
        from .geom import Rectangles
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

        # Handle invert_xaxis/invert_yaxis
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        
        xsel, ysel = xcats or (x0, x1), ycats or (y0, y1)
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
            other = tuple(dim(c) for c in index_cols[1:])
            vals = dim(index_cols[0], util.unique_zip, *other).apply(sel)
            selection_expr = dim(index_cols[0], util.lzip, *other).isin(vals).iloc[:, 0]
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


class Selection1DExpr(Selection2DExpr):
    """
    Mixin class for Cartesian 1D Chart elements to add basic support for
    SelectionExpr streams.
    """

    _selection_dims = 1

    _inverted_expr = False

    def _get_selection_expr_for_stream_value(self, **kwargs):
        invert_axes = self.opts.get('plot').kwargs.get('invert_axes', False)
        region_el = HSpan if invert_axes or self._inverted_expr else VSpan
        if kwargs.get('bounds', None) is None:
            region = None if 'index_cols' in kwargs else NdOverlay({0: region_el()})
            return None, None, region

        x0, y0, x1, y1 = kwargs['bounds']

        # Handle invert_xaxis/invert_yaxis
        if y0 > y1:
            y0, y1 = y1, y0
        if x0 > x1:
            x0, x1 = x1, x0

        if len(self.dimensions()) == 1:
            xdim = self.dimensions()[0]
            ydim = None
        else:
            xdim, ydim = self.dimensions()[:2]

        if invert_axes:
            x0, x1, y0, y1 = y0, y1, x0, x1
            xdim, ydim = ydim, xdim
            cat_kwarg = 'y_selection'
        else:
            cat_kwarg = 'x_selection'
            
        if self._inverted_expr:
            if ydim is not None: xdim = ydim
            x0, x1 = y0, y1
            cat_kwarg = ('y' if invert_axes else 'x') + '_selection'
        cats = kwargs.get(cat_kwarg)

        bbox = {xdim.name: (x0, x1)}
        if cats is not None and len(self.kdims) == 1:
            bbox[self.kdims[0].name] = cats
        index_cols = kwargs.get('index_cols')
        if index_cols:
            index_cols = [self.get_dimension(c) for c in index_cols]
            sel = self.dataset.select(**bbox)
            vals = dim(index_cols[0], util.unique_zip, *index_cols[1:]).apply(sel)
            selection_expr = dim(
                index_cols[0], util.lzip, *index_cols[1:]
            ).isin(vals).iloc[:, 0]
            region_element = None
        else:
            if cats and xdim is self.kdims[0]:
                selection_expr = dim(xdim).isin(cats)
            else:
                selection_expr = ((dim(xdim) >= x0) & (dim(xdim) <= x1))
                if cats is not None and len(self.kdims) == 1:
                    selection_expr &= dim(self.kdims[0]).isin(cats)
            region_element = NdOverlay({0: region_el(x0, x1)})
        return selection_expr, bbox, region_element

    @staticmethod
    def _merge_regions(region1, region2, operation):
        if region1 is None or operation == "overwrite":
            return region2
        data = [d.data for d in region1] + [d.data for d in region2]
        prev = len(data)
        new = None
        while prev != new:
            prev = len(data)
            contiguous = []
            for l, u in data:
                if not util.isfinite(l) or not util.isfinite(u):
                    continue
                overlap = False
                for i, (pl, pu) in enumerate(contiguous):
                    if l >= pl and l <= pu:
                        pu = max(u, pu)
                        overlap = True
                    elif u <= pu and u >= pl:
                        pl = min(l, pl)
                        overlap = True
                    if overlap:
                        contiguous[i] = (pl, pu)
                if not overlap:
                    contiguous.append((l, u))
            new = len(contiguous)
            data = contiguous
        return NdOverlay([(i, region1.last.clone(l, u)) for i, (l, u) in enumerate(data)])
