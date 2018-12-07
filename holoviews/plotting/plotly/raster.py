from __future__ import absolute_import, division, unicode_literals

import numpy as np

from ...core.options import SkipRendering
from ...element import Image, Raster
from .element import ColorbarPlot


class RasterPlot(ColorbarPlot):

    style_opts = ['cmap', 'alpha']

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
        return [dict(x0=l+dx/2., y0=b+dy/2., dx=dx, dy=dy, z=array)]


class HeatMapPlot(RasterPlot):

    def get_extents(self, element, ranges, range_type='combined'):
        return (np.NaN,)*4

    def init_layout(self, key, element, ranges):
        layout = super(HeatMapPlot, self).init_layout(key, element, ranges)
        gridded = element.gridded
        xlabels, ylabels = (gridded.dimension_values(i, False) for i in range(2))
        xvals = np.arange(len(xlabels))
        yvals = np.arange(len(ylabels))
        layout['xaxis']['tickvals'] = xvals
        layout['xaxis']['ticktext'] = xlabels
        layout['yaxis']['tickvals'] = xvals
        layout['yaxis']['ticktext'] = xlabels
        return layout

    def get_data(self, element, ranges, style):
        gridded = element.gridded
        yn, xn = gridded.interface.shape(gridded, True)
        return [dict(x=np.arange(xn), y=np.arange(yn),
                     z=gridded.dimension_values(2, flat=False))]


class QuadMeshPlot(RasterPlot):

    def get_data(self, element, ranges, style):
        x, y, z = element.dimensions()[:3]
        irregular = element.interface.irregular(element, x)
        if irregular:
            raise SkipRendering("Plotly QuadMeshPlot only supports rectilinear meshes")
        xc, yc = (element.interface.coords(element, x, edges=True, ordered=True),
                  element.interface.coords(element, y, edges=True, ordered=True))
        zdata = element.dimension_values(z, flat=False)
        return [dict(x=xc, y=yc, z=zdata)]
