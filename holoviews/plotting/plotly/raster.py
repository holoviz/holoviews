from __future__ import absolute_import, division, unicode_literals

import numpy as np

from ...core.options import SkipRendering
from ...element import Image, Raster
from .element import ColorbarPlot


class RasterPlot(ColorbarPlot):

    style_opts = ['cmap']

    trace_kwargs = {'type': 'heatmap'}

    def graph_options(self, element, ranges, style):
        opts = super(RasterPlot, self).graph_options(element, ranges, style)
        copts = self.get_color_opts(element.vdims[0], element, ranges, style)
        opts['zmin'] = copts.pop('cmin')
        opts['zmax'] = copts.pop('cmax')
        opts['zauto'] = copts.pop('cauto')
        return dict(opts, **copts)

    def get_data(self, element, ranges, style):
        if isinstance(element, Image):
            l, b, r, t = element.bounds.lbrt()
        else:
            l, b, r, t = element.extents
        array = element.dimension_values(2, flat=False)
        if type(element) is Raster:
            array=array.T[::-1,...]
        ny, nx = array.shape
        dx, dy = float(r-l)/nx, float(t-b)/ny
        return (), dict(x0=l, y0=b, dx=dx, dy=dy, z=array)


class HeatMapPlot(RasterPlot):

    def get_extents(self, element, ranges, range_type='combined'):
        return (np.NaN,)*4

    def get_data(self, element, ranges, style):
        gridded = element.gridded.sort()
        return (), dict(x=gridded.dimension_values(0, False),
                        y=gridded.dimension_values(1, False),
                        z=gridded.dimension_values(2, flat=False))


class QuadMeshPlot(RasterPlot):

    def get_data(self, element, ranges, style):
        if len(set(v.shape for v in element.data)) == 1:
            raise SkipRendering("Plotly QuadMeshPlot only supports rectangular meshes")
        return (), dict(x=element.data[0], y=element.data[1],
                        z=element.data[2])
