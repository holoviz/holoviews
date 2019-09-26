from __future__ import absolute_import, division, unicode_literals

import param

from .selection import PlotlyOverlaySelectionDisplay
from .chart import ChartPlot
from .element import ElementPlot, ColorbarPlot


class BivariatePlot(ChartPlot, ColorbarPlot):

    filled = param.Boolean(default=False)

    ncontours = param.Integer(default=None)

    trace_kwargs = {'type': 'histogram2dcontour'}

    style_opts = ['visible', 'cmap', 'showlabels', 'labelfont', 'labelformat', 'showlines']

    _style_key = 'contours'

    selection_display = PlotlyOverlaySelectionDisplay()

    def graph_options(self, element, ranges, style):
        opts = super(BivariatePlot, self).graph_options(element, ranges, style)
        copts = self.get_color_opts(element.vdims[0], element, ranges, style)

        if self.ncontours:
            opts['autocontour'] = False
            opts['ncontours'] = self.ncontours

        # Make line width a little wider (default is less than 1)
        opts['line'] = {'width': 1}

        # Configure contours
        opts['contours'] = {
            'coloring': 'fill' if self.filled else 'lines',
            'showlines': style.get('showlines', True)
        }

        # Add colorscale
        opts['colorscale'] = copts['colorscale']

        # Add colorbar
        if 'colorbar' in copts:
            opts['colorbar'] = copts['colorbar']

        opts['showscale'] = copts.get('showscale', False)

        # Add visible
        opts['visible'] = style.get('visible', True)

        return opts


class DistributionPlot(ElementPlot):

    bandwidth = param.Number(default=None, doc="""
        The bandwidth of the kernel for the density estimate.""")

    cut = param.Number(default=3, doc="""
        Draw the estimate to cut * bw from the extreme data points.""")

    filled = param.Boolean(default=True, doc="""
        Whether the bivariate contours should be filled.""")

    style_opts = ['visible', 'color', 'dash', 'line_width']

    trace_kwargs = {'type': 'scatter', 'mode': 'lines'}

    _style_key = 'line'

    selection_display = PlotlyOverlaySelectionDisplay()


class MultiDistributionPlot(ElementPlot):

    def _get_axis_dims(self, element):
        return element.kdims, element.vdims[0]

    def get_data(self, element, ranges, style):
        if element.kdims:
            groups = element.groupby(element.kdims).items()
        else:
            groups = [(element.label, element)]
        plots = []
        axis = 'x' if self.invert_axes else 'y'
        for key, group in groups:
            if element.kdims:
                label = ','.join([d.pprint_value(v) for d, v in zip(element.kdims, key)])
            else:
                label = key
            data = {axis: group.dimension_values(group.vdims[0]), 'name': label}
            plots.append(data)
        return plots

    def get_extents(self, element, ranges, range_type='combined'):
        return super(MultiDistributionPlot, self).get_extents(
            element, ranges, range_type, 'categorical', element.vdims[0]
        )


class BoxWhiskerPlot(MultiDistributionPlot):

    boxpoints = param.ObjectSelector(objects=["all", "outliers",
                                              "suspectedoutliers", False],
                                     default='outliers', doc="""
        Which points to show, valid options are 'all', 'outliers',
        'suspectedoutliers' and False""")

    jitter = param.Number(default=0, doc="""
        Sets the amount of jitter in the sample points drawn. If "0",
        the sample points align along the distribution axis. If "1",
        the sample points are drawn in a random jitter of width equal
        to the width of the box(es).""")

    mean = param.ObjectSelector(default=False, objects=[True, False, 'sd'],
                                doc="""
        If "True", the mean of the box(es)' underlying distribution
        is drawn as a dashed line inside the box(es). If "sd" the
        standard deviation is also drawn.""")

    style_opts = ['visible', 'color', 'alpha', 'outliercolor', 'marker', 'size']

    trace_kwargs = {'type': 'box'}

    _style_key = 'marker'

    selection_display = PlotlyOverlaySelectionDisplay()

    def graph_options(self, element, ranges, style):
        options = super(BoxWhiskerPlot, self).graph_options(element, ranges, style)
        options['boxmean'] = self.mean
        options['jitter'] = self.jitter
        return options


class ViolinPlot(MultiDistributionPlot):


    box = param.Boolean(default=True, doc="""
        Whether to draw a boxplot inside the violin""")

    meanline = param.Boolean(default=False, doc="""
        If "True", the mean of the box(es)' underlying distribution
        is drawn as a dashed line inside the box(es). If "sd" the
        standard deviation is also drawn.""")

    style_opts = ['visible', 'color', 'alpha', 'outliercolor', 'marker', 'size']

    trace_kwargs = {'type': 'violin'}

    _style_key = 'marker'

    def graph_options(self, element, ranges, style):
        options = super(ViolinPlot, self).graph_options(element, ranges, style)
        options['meanline'] = {'visible': self.meanline}
        options['box'] = {'visible': self.box}
        return options
