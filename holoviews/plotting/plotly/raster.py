import numpy as np
import plotly.graph_objs as go

from ...core.options import SkipRendering
from ...element import Image, Raster
from .element import ColorbarPlot


class RasterPlot(ColorbarPlot):

    style_opts = ['cmap']

    graph_obj = go.Heatmap

    def graph_options(self, element, ranges):
        opts = super(RasterPlot, self).graph_options(element, ranges)
        style = self.style[self.cyclic_index]
        copts = self.get_color_opts(element.vdims[0], element, ranges, style)
        opts['zmin'] = copts.pop('cmin')
        opts['zmax'] = copts.pop('cmax')
        opts['zauto'] = copts.pop('cauto')
        return dict(opts, **copts)

    def get_data(self, element, ranges):
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

    def get_extents(self, element, ranges):
        return (np.NaN,)*4

    def get_data(self, element, ranges):
        gridded = element.gridded.sort()
        return (), dict(x=gridded.dimension_values(0, False),
                        y=gridded.dimension_values(1, False),
                        z=gridded.dimension_values(2, flat=False))


class QuadMeshPlot(RasterPlot):

    def get_data(self, element, ranges):
        if len(set(v.shape for v in element.data)) == 1:
            raise SkipRendering("Plotly QuadMeshPlot only supports rectangular meshes")
        return (), dict(x=element.data[0], y=element.data[1],
                        z=element.data[2])
