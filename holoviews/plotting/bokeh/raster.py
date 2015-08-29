from itertools import product
import numpy as np

from bokeh.models.mappers import LinearColorMapper

from ...element import Image, RGB, Raster
from .element import ElementPlot, line_properties, fill_properties
from .util import mplcmap_to_palette, map_colors, get_cmap


class RasterPlot(ElementPlot):

    style_opts = ['cmap']
    _plot_method = 'image'

    def __init__(self, *args, **kwargs):
        super(RasterPlot, self).__init__(*args, **kwargs)
        if self.map.type == Raster:
            self.invert_yaxis = not self.invert_yaxis


    def get_data(self, element, ranges=None):
        img = element.data
        if isinstance(element, Image):
            l, b, r, t = element.bounds.lbrt()
        else:
            l, b, r, t = element.extents
        dh = t-b
        if type(element) is Raster:
            b = t
        mapping = dict(image='image', x='x', y='y', dw='dw', dh='dh')
        return (dict(image=[np.flipud(img)], x=[l],
                     y=[b], dw=[r-l], dh=[dh]), mapping)


    def _glyph_properties(self, plot, element, source, ranges):
        properties = super(RasterPlot, self)._glyph_properties(plot, element,
                                                               source, ranges)
        properties = {k: v for k, v in properties.items()}
        val_dim = [d.name for d in element.vdims][0]
        low, high = ranges.get(val_dim)
        if 'cmap' in properties:
            palette = mplcmap_to_palette(properties.pop('cmap', None))
        properties['color_mapper'] = LinearColorMapper(palette, low=low, high=high)
        return properties


class RGBPlot(RasterPlot):

    style_opts = []
    _plot_method = 'image_rgba'

    def get_data(self, element, ranges=None):
        data, mapping = super(RGBPlot, self).get_data(element, ranges)
        img = data['image'][0]
        if img.ndim == 3:
            if img.shape[2] == 3: # alpha channel not included
                img = np.dstack([img, np.ones(img.shape[:2])])
            img = (img*255).astype(np.uint8)
            N, M, _ = img.shape
            #convert image NxM dtype=uint32
            img = img.view(dtype=np.uint32).reshape((N, M))
            data['image'] = [img]
        return data, mapping

    def _glyph_properties(self, plot, element, source, ranges):
        return ElementPlot._glyph_properties(self, plot, element,
                                             source, ranges)


class HeatmapPlot(ElementPlot):

    _plot_method = 'rect'
    style_opts = ['cmap', 'color'] + line_properties + fill_properties

    def _init_axes(self, plots, element, ranges):
        labels = self._axis_labels(element, plots)
        xvals, yvals = element.dense_keys()
        plot_ranges = {'x_range': [str(x) for x in xvals],
                       'y_range': [str(y) for y in yvals]}
        return ('auto', 'auto'), labels, plot_ranges


    def get_data(self, element, ranges=None):
        style = self.style[self.cyclic_index]
        cmap = style.get('palette', style.get('cmap', None))
        cmap = get_cmap(cmap)
        x, y, z = element.dimensions(label=True)
        zvals = np.rot90(element.data, 3).flatten()
        colors = map_colors(zvals, ranges[z], cmap)
        xvals, yvals = zip(*product(*element.dense_keys()))
        return ({x: xvals, y: yvals, z: zvals, 'color': colors},
                {'x': x, 'y': y, 'fill_color': 'color', 'height': 1, 'width': 1})
