import numpy as np

from bokeh.models.mappers import LinearColorMapper

from ...element import Image, RGB, Raster
from .element import ElementPlot
from .util import mplcmap_to_palette


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
