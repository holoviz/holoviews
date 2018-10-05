import numpy as np
import param

from ...core.util import cartesian_product, dimension_sanitizer, isfinite
from ...element import Raster, RGB, HSV
from .element import ElementPlot, ColorbarPlot, line_properties, fill_properties
from .util import mpl_to_bokeh, colormesh, bokeh_version


class RasterPlot(ColorbarPlot):

    clipping_colors = param.Dict(default={'NaN': 'transparent'})

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['cmap', 'alpha']
    _plot_methods = dict(single='image')

    def _hover_opts(self, element):
        xdim, ydim = element.kdims
        tooltips = [(xdim.pprint_label, '$x'), (ydim.pprint_label, '$y')]
        if bokeh_version >= '0.12.16' and not isinstance(element, (RGB, HSV)):
            vdim = element.vdims[0]
            tooltips.append((vdim.pprint_label, '@image'))
        return tooltips, {}

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

        if 0 in img.shape:
            img = np.array([[np.NaN]])

        data = dict(image=[img], x=[l], y=[b], dw=[dw], dh=[dh])
        return (data, mapping, style)



class RGBPlot(RasterPlot):

    style_opts = ['alpha']
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

        if 0 in img.shape:
            img = np.zeros((1, 1), dtype=np.uint32)

        data = dict(image=[img], x=[l], y=[b], dw=[dw], dh=[dh])
        return (data, mapping, style)

    def _glyph_properties(self, plot, element, source, ranges, style):
        return ElementPlot._glyph_properties(self, plot, element,
                                             source, ranges, style)



class HSVPlot(RGBPlot):

    def get_data(self, element, ranges, style):
        return super(HSVPlot, self).get_data(element.rgb, ranges, style)



class QuadMeshPlot(ColorbarPlot):

    clipping_colors = param.Dict(default={'NaN': 'transparent'})

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    _plot_methods = dict(single='quad')
    style_opts = ['cmap', 'color'] + line_properties + fill_properties

    def get_data(self, element, ranges, style):
        x, y, z = element.dimensions()[:3]
        if self.invert_axes: x, y = y, x
        cmapper = self._get_colormapper(z, element, ranges, style)
        cmapper = {'field': z.name, 'transform': cmapper}

        irregular = element.interface.irregular(element, x)
        if irregular:
            mapping = dict(xs='xs', ys='ys', fill_color=cmapper)
        else:
            mapping = {'left': 'left', 'right': 'right',
                       'fill_color': cmapper,
                       'top': 'top', 'bottom': 'bottom'}

        if self.static_source:
            return {}, mapping, style

        x, y = dimension_sanitizer(x.name), dimension_sanitizer(y.name)

        zdata = element.dimension_values(z, flat=False)
        if irregular:
            dims = element.kdims
            if self.invert_axes: dims = dims[::-1]
            X, Y = [element.interface.coords(element, d, expanded=True, edges=True)
                    for d in dims]
            X, Y = colormesh(X, Y)
            zvals = zdata.T.flatten() if self.invert_axes else zdata.flatten()
            XS, YS = [], []
            mask = []
            xc, yc = [], []
            for xs, ys, zval in zip(X, Y, zvals):
                xs, ys = xs[:-1], ys[:-1]
                if isfinite(zval) and all(isfinite(xs)) and all(isfinite(ys)):
                    XS.append(list(xs))
                    YS.append(list(ys))
                    mask.append(True)
                    if 'hover' in self.handles:
                        xc.append(xs.mean())
                        yc.append(ys.mean())
                else:
                    mask.append(False)

            data = {'xs': XS, 'ys': YS, z.name: zvals[np.array(mask)]}
            if 'hover' in self.handles:
                data[x] = np.array(xc)
                data[y] = np.array(yc)
        else:
            xc, yc = (element.interface.coords(element, x, edges=True, ordered=True),
                      element.interface.coords(element, y, edges=True, ordered=True))
            
            x0, y0 = cartesian_product([xc[:-1], yc[:-1]], copy=True)
            x1, y1 = cartesian_product([xc[1:], yc[1:]], copy=True)
            zvals = zdata.flatten() if self.invert_axes else zdata.T.flatten()
            data = {'left': x0, 'right': x1, dimension_sanitizer(z.name): zvals,
                    'bottom': y0, 'top': y1}
            if 'hover' in self.handles and not self.static_source:
                data[x] = element.dimension_values(x)
                data[y] = element.dimension_values(y)
        return data, mapping, style


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        properties = dict(properties, **mapping)
        if 'xs' in mapping:
            renderer = plot.patches(**properties)
        else:
            renderer = plot.quad(**properties)
        if self.colorbar and 'color_mapper' in self.handles:
            self._draw_colorbar(plot, self.handles['color_mapper'])
        return renderer, renderer.glyph
