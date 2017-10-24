import numpy as np
import param

from bokeh.models import HoverTool
from ...core.util import cartesian_product, is_nan, dimension_sanitizer
from ...element import Image, Raster
from ..renderer import SkipRendering
from .element import ElementPlot, ColorbarPlot, line_properties, fill_properties


class RasterPlot(ColorbarPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['cmap']
    _plot_methods = dict(single='image')

    def __init__(self, *args, **kwargs):
        super(RasterPlot, self).__init__(*args, **kwargs)
        if self.hmap.type == Raster:
            self.invert_yaxis = not self.invert_yaxis

    def get_data(self, element, ranges, style):
        mapping = dict(image='image', x='x', y='y', dw='dw', dh='dh')
        val_dim = [d for d in element.vdims][0]
        style['color_mapper'] = self._get_colormapper(val_dim, element, ranges, style)

        if self.static_source:
            return {}, mapping, style

        img = element.dimension_values(2, flat=False)
        if img.dtype.kind == 'b':
            img = img.astype(np.int8)

        if type(element) is Raster:
            l, b, r, t = element.extents
            if self.invert_axes:
                l, b, r, t = b, l, t, r
            else:
                img = img.T
        else:
            l, b, r, t = element.bounds.lbrt()
            if self.invert_axes:
                img = img.T
                l, b, r, t = b, l, t, r

        if self.invert_xaxis:
            l, r = r, l
            img = img[:, ::-1]
        if self.invert_yaxis:
            img = img[::-1]
            b, t = t, b
        dh, dw = t-b, r-l

        data = dict(image=[img], x=[l], y=[b], dw=[dw], dh=[dh])
        return (data, mapping, style)



class RGBPlot(RasterPlot):

    style_opts = []
    _plot_methods = dict(single='image_rgba')

    def get_data(self, element, ranges, style):
        mapping = dict(image='image', x='x', y='y', dw='dw', dh='dh')
        if self.static_source:
            return {}, mapping, style

        img = np.dstack([element.dimension_values(d, flat=False)
                         for d in element.vdims])
        if img.ndim == 3:
            if img.shape[2] == 3: # alpha channel not included
                alpha = np.ones(img.shape[:2])
                if img.dtype.name == 'uint8':
                    alpha = (alpha*255).astype('uint8')
                img = np.dstack([img, alpha])
            if img.dtype.name != 'uint8':
                img = (img*255).astype(np.uint8)
            N, M, _ = img.shape
            #convert image NxM dtype=uint32
            img = img.view(dtype=np.uint32).reshape((N, M))

        # Ensure axis inversions are handled correctly
        l, b, r, t = element.bounds.lbrt()
        if self.invert_axes:
            img = img.T
            l, b, r, t = b, l, t, r
        if self.invert_xaxis:
            l, r = r, l
            img = img[:, ::-1]
        if self.invert_yaxis:
            img = img[::-1]
            b, t = t, b
        dh, dw = t-b, r-l

        data = dict(image=[img], x=[l], y=[b], dw=[dw], dh=[dh])
        return (data, mapping, style)

    def _glyph_properties(self, plot, element, source, ranges, style):
        return ElementPlot._glyph_properties(self, plot, element,
                                             source, ranges, style)

class HSVPlot(RGBPlot):

    def get_data(self, element, ranges, style):
        return super(HSVPlot, self).get_data(element.rgb, ranges, style)


class HeatMapPlot(ColorbarPlot):

    clipping_colors = param.Dict(default={'NaN': 'white'}, doc="""
        Dictionary to specify colors for clipped values, allows
        setting color for NaN values and for values above and below
        the min and max value. The min, max or NaN color may specify
        an RGB(A) color as a color hex string of the form #FFFFFF or
        #FFFFFFFF or a length 3 or length 4 tuple specifying values in
        the range 0-1 or a named HTML color.""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    _plot_methods = dict(single='rect')
    style_opts = ['cmap', 'color'] + line_properties + fill_properties

    _categorical = True

    def _get_factors(self, element):
        return super(HeatMapPlot, self)._get_factors(element.gridded)

    def get_data(self, element, ranges, style):
        x, y, z = [dimension_sanitizer(d) for d in element.dimensions(label=True)[:3]]
        if self.invert_axes: x, y = y, x
        cmapper = self._get_colormapper(element.vdims[0], element, ranges, style)
        if self.static_source:
            return {}, {'x': x, 'y': y, 'fill_color': {'field': 'zvalues', 'transform': cmapper}}, style

        aggregate = element.gridded
        xdim, ydim = aggregate.dimensions()[:2]
        xvals, yvals = (aggregate.dimension_values(x),
                        aggregate.dimension_values(y))
        zvals = aggregate.dimension_values(2, flat=False)
        if self.invert_axes:
            xdim, ydim = ydim, xdim
            zvals = zvals.T.flatten()
        else:
            zvals = zvals.T.flatten()
        if xvals.dtype.kind not in 'SU':
            xvals = [xdim.pprint_value(xv) for xv in xvals]
        if yvals.dtype.kind not in 'SU':
            yvals = [ydim.pprint_value(yv) for yv in yvals]
        data = {x: xvals, y: yvals, 'zvalues': zvals}

        if any(isinstance(t, HoverTool) for t in self.state.tools) and not self.static_source:
            for vdim in element.vdims:
                sanitized = dimension_sanitizer(vdim.name)
                data[sanitized] = ['-' if is_nan(v) else vdim.pprint_value(v)
                                   for v in aggregate.dimension_values(vdim)]
        return (data, {'x': x, 'y': y, 'fill_color': {'field': 'zvalues', 'transform': cmapper},
                       'height': 1, 'width': 1}, style)


class QuadMeshPlot(ColorbarPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    _plot_methods = dict(single='rect')
    style_opts = ['cmap', 'color'] + line_properties + fill_properties

    def get_data(self, element, ranges, style):
        x, y, z = element.dimensions(label=True)
        if self.invert_axes: x, y = y, x
        cmapper = self._get_colormapper(element.vdims[0], element, ranges, style)
        if self.static_source:
            return {}, {'x': x, 'y': y, 'fill_color': {'field': z, 'transform': cmapper}}, style

        if len(set(v.shape for v in element.data)) == 1:
            raise SkipRendering("Bokeh QuadMeshPlot only supports rectangular meshes")
        zdata = element.data[2]
        xvals = element.dimension_values(0, False)
        yvals = element.dimension_values(1, False)
        widths = np.diff(element.data[0])
        heights = np.diff(element.data[1])
        if self.invert_axes:
            zvals = zdata.flatten()
            xvals, yvals, widths, heights = yvals, xvals, heights, widths
        else:
            zvals = zdata.T.flatten()
        xs, ys = cartesian_product([xvals, yvals], copy=True)
        ws, hs = cartesian_product([widths, heights], copy=True)
        data = {x: xs, y: ys, z: zvals, 'widths': ws, 'heights': hs}
        return (data, {'x': x, 'y': y,
                       'fill_color': {'field': z, 'transform': cmapper},
                       'height': 'heights', 'width': 'widths'}, style)
