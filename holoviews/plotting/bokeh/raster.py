from __future__ import absolute_import, division, unicode_literals

import sys

import numpy as np
import param

from bokeh.models import DatetimeAxis, CustomJSHover

from ...core.util import cartesian_product, dimension_sanitizer, isfinite
from ...element import Raster
from ..util import categorical_legend
from .chart import PointPlot
from .element import ColorbarPlot, LegendPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, mpl_to_bokeh
from .util import colormesh


class RasterPlot(ColorbarPlot):

    clipping_colors = param.Dict(default={'NaN': 'transparent'})

    nodata = param.Integer(default=None, doc="""
        Optional missing-data value for integer data.
        If non-None, data with this value will be replaced with NaN so
        that it is transparent (by default) when plotted.""")

    padding = param.ClassSelector(default=0, class_=(int, float, tuple))

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = base_properties + ['cmap', 'alpha']

    _nonvectorized_styles = style_opts

    _plot_methods = dict(single='image')

    selection_display = BokehOverlaySelectionDisplay()

    def _hover_opts(self, element):
        xdim, ydim = element.kdims
        tooltips = [(xdim.pprint_label, '$x'), (ydim.pprint_label, '$y')]
        vdims = element.vdims
        tooltips.append((vdims[0].pprint_label, '@image'))
        for vdim in vdims[1:]:
            vname = dimension_sanitizer(vdim.name)
            tooltips.append((vdim.pprint_label, '@{0}'.format(vname)))
        return tooltips, {}

    def _postprocess_hover(self, renderer, source):
        super()._postprocess_hover(renderer, source)
        hover = self.handles.get('hover')
        if not (hover and isinstance(hover.tooltips, list)):
            return

        element = self.current_frame
        xdim, ydim = [dimension_sanitizer(kd.name) for kd in element.kdims]
        xaxis = self.handles['xaxis']
        yaxis = self.handles['yaxis']

        code = """
        var {ax} = special_vars.{ax};
        var date = new Date({ax});
        return date.toISOString().slice(0, 19).replace('T', ' ')
        """
        tooltips, formatters = [], dict(hover.formatters)
        for (name, formatter) in hover.tooltips:
            if isinstance(xaxis, DatetimeAxis) and formatter == '$x':
                xhover = CustomJSHover(code=code.format(ax='x'))
                formatters['$x'] = xhover
                formatter += '{custom}'
            if isinstance(yaxis, DatetimeAxis) and formatter == '$y':
                yhover = CustomJSHover(code=code.format(ax='y'))
                formatters['$y'] = yhover
                formatter += '{custom}'
            tooltips.append((name, formatter))
        hover.tooltips = tooltips
        hover.formatters = formatters

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hmap.type == Raster:
            self.invert_yaxis = not self.invert_yaxis

    def get_data(self, element, ranges, style):
        mapping = dict(image='image', x='x', y='y', dw='dw', dh='dh')
        val_dim = element.vdims[0]
        style['color_mapper'] = self._get_colormapper(val_dim, element, ranges, style)
        if 'alpha' in style:
            style['global_alpha'] = style['alpha']

        if self.static_source:
            return {}, mapping, style

        if type(element) is Raster:
            l, b, r, t = element.extents
            if self.invert_axes:
                l, b, r, t = b, l, t, r
        else:
            l, b, r, t = element.bounds.lbrt()
            if self.invert_axes:
                l, b, r, t = b, l, t, r

        dh, dw = t-b, r-l
        if self.invert_xaxis:
            l, r = r, l
        if self.invert_yaxis:
            b, t = t, b
        data = dict(x=[l], y=[b], dw=[dw], dh=[dh])

        for i, vdim in enumerate(element.vdims, 2):
            if i > 2 and 'hover' not in self.handles:
                break
            img = element.dimension_values(i, flat=False)
            if img.dtype.kind == 'b':
                img = img.astype(np.int8)
            if 0 in img.shape:
                img = np.array([[np.NaN]])
            if ((self.invert_axes and not type(element) is Raster) or
                (not self.invert_axes and type(element) is Raster)):
                img = img.T
            if self.invert_xaxis:
                img = img[:, ::-1]
            if self.invert_yaxis:
                img = img[::-1]
            key = 'image' if i == 2 else dimension_sanitizer(vdim.name)
            data[key] = [img]

        return (data, mapping, style)



class RGBPlot(LegendPlot):

    padding = param.ClassSelector(default=0, class_=(int, float, tuple))

    style_opts = ['alpha'] + base_properties

    _nonvectorized_styles = style_opts

    _plot_methods = dict(single='image_rgba')

    selection_display = BokehOverlaySelectionDisplay()

    def __init__(self, hmap, **params):
        super(RGBPlot, self).__init__(hmap, **params)
        self._legend_plot = None

    def _hover_opts(self, element):
        xdim, ydim = element.kdims
        return [(xdim.pprint_label, '$x'), (ydim.pprint_label, '$y'),
                ('RGBA', '@image')], {}

    def _init_glyphs(self, plot, element, ranges, source):
        super(RGBPlot, self)._init_glyphs(plot, element, ranges, source)
        if not ('holoviews.operation.datashader' in sys.modules and self.show_legend):
            return
        try:
            legend = categorical_legend(element, backend=self.backend)
        except Exception:
            return
        if legend is None:
            return
        legend_params = {k: v for k, v in self.param.get_param_values()
                         if k.startswith('legend')}
        self._legend_plot = PointPlot(legend, keys=[], overlaid=1, **legend_params)
        self._legend_plot.initialize_plot(plot=plot)
        self._legend_plot.handles['glyph_renderer'].tags.append('hv_legend')
        self.handles['rgb_color_mapper'] = self._legend_plot.handles['color_color_mapper']

    def get_data(self, element, ranges, style):
        mapping = dict(image='image', x='x', y='y', dw='dw', dh='dh')
        if 'alpha' in style:
            style['global_alpha'] = style['alpha']

        if self.static_source:
            return {}, mapping, style

        img = np.dstack([element.dimension_values(d, flat=False)
                         for d in element.vdims])
        if img.ndim == 3:
            if img.dtype.kind == 'f':
                img = img*255
            if img.size and (img.min() < 0 or img.max() > 255):
                self.param.warning('Clipping input data to the valid '
                                   'range for RGB data ([0..1] for '
                                   'floats or [0..255] for integers).')
                img = np.clip(img, 0, 255)

            if img.dtype.name != 'uint8':
                img = img.astype(np.uint8)
            if img.shape[2] == 3: # alpha channel not included
                alpha = np.full(img.shape[:2], 255, dtype='uint8')
                img = np.dstack([img, alpha])
            N, M, _ = img.shape
            #convert image NxM dtype=uint32
            if not img.flags['C_CONTIGUOUS']:
                img = img.copy()
            img = img.view(dtype=np.uint32).reshape((N, M))

        # Ensure axis inversions are handled correctly
        l, b, r, t = element.bounds.lbrt()
        if self.invert_axes:
            img = img.T
            l, b, r, t = b, l, t, r

        dh, dw = t-b, r-l
        if self.invert_xaxis:
            l, r = r, l
            img = img[:, ::-1]
        if self.invert_yaxis:
            img = img[::-1]
            b, t = t, b

        if 0 in img.shape:
            img = np.zeros((1, 1), dtype=np.uint32)

        data = dict(image=[img], x=[l], y=[b], dw=[dw], dh=[dh])
        return (data, mapping, style)


class HSVPlot(RGBPlot):

    def get_data(self, element, ranges, style):
        return super().get_data(element.rgb, ranges, style)


class QuadMeshPlot(ColorbarPlot):

    clipping_colors = param.Dict(default={'NaN': 'transparent'})

    nodata = param.Integer(default=None, doc="""
        Optional missing-data value for integer data.
        If non-None, data with this value will be replaced with NaN so
        that it is transparent (by default) when plotted.""")

    padding = param.ClassSelector(default=0, class_=(int, float, tuple))

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    selection_display = BokehOverlaySelectionDisplay()

    style_opts = ['cmap'] + base_properties + line_properties + fill_properties

    _nonvectorized_styles = style_opts

    _plot_methods = dict(single='quad')

    def get_data(self, element, ranges, style):
        x, y, z = element.dimensions()[:3]

        if self.invert_axes: x, y = y, x
        cmapper = self._get_colormapper(z, element, ranges, style)
        cmapper = {'field': z.name, 'transform': cmapper}

        irregular = (element.interface.irregular(element, x) or
                     element.interface.irregular(element, y))
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
                hover_dims = element.dimensions()[3:]
                hover_data = [element.dimension_values(hover_dim, flat=False)
                              for hover_dim in hover_dims]
                for hdim, hdat in zip(hover_dims, hover_data):
                    data[dimension_sanitizer(hdim.name)] = (hdat.flatten()
                        if self.invert_axes else hdat.T.flatten())
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
