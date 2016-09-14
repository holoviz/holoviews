import numpy as np
import param

from ...core.util import cartesian_product
from ...element import Image, Raster, RGB
from ..renderer import SkipRendering
from ..util import map_colors
from .element import ElementPlot, ColorbarPlot, line_properties, fill_properties
from .util import mplcmap_to_palette, get_cmap, hsv_to_rgb, mpl_to_bokeh


class RasterPlot(ColorbarPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['cmap']
    _plot_methods = dict(single='image')
    _update_handles = ['color_mapper', 'source', 'glyph']

    def __init__(self, *args, **kwargs):
        super(RasterPlot, self).__init__(*args, **kwargs)
        if self.hmap.type == Raster:
            self.invert_yaxis = not self.invert_yaxis


    def get_data(self, element, ranges=None, empty=False):
        img = element.data
        if isinstance(element, Image):
            l, b, r, t = element.bounds.lbrt()
        else:
            l, b, r, t = element.extents
        dh = t-b
        if type(element) is Raster:
            b = t

        mapping = dict(image='image', x='x', y='y', dw='dw', dh='dh')
        if empty:
            data = dict(image=[], x=[], y=[], dw=[], dh=[])
        else:
            data = dict(image=[np.flipud(img)], x=[l],
                        y=[b], dw=[r-l], dh=[dh])
        return (data, mapping)


    def _glyph_properties(self, plot, element, source, ranges):
        properties = super(RasterPlot, self)._glyph_properties(plot, element,
                                                               source, ranges)
        properties = {k: v for k, v in properties.items()}
        val_dim = [d for d in element.vdims][0]
        properties['color_mapper'] = self._get_colormapper(val_dim, element, ranges,
                                                           properties)
        return properties



class ImagePlot(RasterPlot):

    def get_data(self, element, ranges=None, empty=False):
        img = element.dimension_values(2, flat=False)
        l, b, r, t = element.bounds.lbrt()
        dh, dw = t-b, r-l
        mapping = dict(image='image', x='x', y='y', dw='dw', dh='dh')
        if empty:
            data = dict(image=[], x=[], y=[], dw=[], dh=[])
        else:
            data = dict(image=[img], x=[l],
                        y=[b], dw=[dw], dh=[dh])
        return (data, mapping)


class RGBPlot(RasterPlot):

    style_opts = []
    _plot_methods = dict(single='image_rgba')

    def get_data(self, element, ranges=None, empty=False):
        data, mapping = super(RGBPlot, self).get_data(element, ranges, empty)
        img = data['image'][0]

        if empty:
            data['image'] = []
        elif img.ndim == 3:
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
            data['image'] = [img]
        return data, mapping


    def _glyph_properties(self, plot, element, source, ranges):
        return ElementPlot._glyph_properties(self, plot, element,
                                             source, ranges)

class HSVPlot(RGBPlot):

    def get_data(self, element, ranges=None, empty=False):
        rgb = RGB(hsv_to_rgb(element.data))
        return super(HSVPlot, self).get_data(rgb, ranges, empty)


class HeatmapPlot(ColorbarPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    _plot_methods = dict(single='rect')
    style_opts = ['cmap', 'color'] + line_properties + fill_properties

    def _axes_props(self, plots, subplots, element, ranges):
        dims = element.dimensions()
        labels = self._get_axis_labels(dims)
        xvals, yvals = [element.dimension_values(i, False)
                        for i in range(2)]
        if self.invert_yaxis: yvals = yvals[::-1]
        plot_ranges = {'x_range': [str(x) for x in xvals],
                       'y_range': [str(y) for y in yvals]}
        return ('auto', 'auto'), labels, plot_ranges


    def get_data(self, element, ranges=None, empty=False):
        x, y, z = element.dimensions(label=True)
        style = self.style[self.cyclic_index]
        cmapper = self._get_colormapper(element.vdims[0], element, ranges, style)
        if empty:
            data = {x: [], y: [], z: [], 'color': []}
        else:
            zvals = np.rot90(element.raster, 3).flatten()
            xvals, yvals = [[str(v) for v in element.dimension_values(i)]
                            for i in range(2)]
            data = {x: xvals, y: yvals, z: zvals}

        return (data, {'x': x, 'y': y, 'fill_color': {'field': z, 'transform': cmapper},
                       'height': 1, 'width': 1})


class QuadMeshPlot(ColorbarPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    _plot_methods = dict(single='rect')
    style_opts = ['cmap', 'color'] + line_properties + fill_properties

    def get_data(self, element, ranges=None, empty=False):
        x, y, z = element.dimensions(label=True)
        style = self.style[self.cyclic_index]
        cmapper = self._get_colormapper(element.vdims[0], element, ranges, style)
        if empty:
            data = {x: [], y: [], z: [], 'height': [], 'width': []}
        else:
            if len(set(v.shape for v in element.data)) == 1:
                raise SkipRendering("Bokeh QuadMeshPlot only supports rectangular meshes")
            zvals = element.data[2].T.flatten()
            xvals = element.dimension_values(0, False)
            yvals = element.dimension_values(1, False)
            widths = np.diff(element.data[0])
            heights = np.diff(element.data[1])
            xs, ys = cartesian_product([xvals, yvals])
            ws, hs = cartesian_product([widths, heights])
            data = {x: xs.flat, y: ys.flat, z: zvals,
                    'widths': ws.flat, 'heights': hs.flat}

        return (data, {'x': x, 'y': y,
                       'fill_color': {'field': z, 'transform': cmapper},
                       'height': 'heights', 'width': 'widths'})
