from functools import partial

import param
import numpy as np

from bokeh.models.ranges import DataRange1d

from ...element import Polygons, Contours, Distribution, Bivariate
from ...operation.stats import univariate_kde, bivariate_kde

from .chart import AreaPlot
from .path import PolygonPlot


class DistributionPlot(AreaPlot):
    """
    DistributionPlot visualizes a distribution of values as a KDE.
    """

    bw = param.Number(default=None)

    def __init__(self, element, plot=None, **params):
        element = element.map(self._convert_element, Distribution)
        super(DistributionPlot, self).__init__(element, plot, **params)

    def _convert_element(self, element):
        plot_opts = self.lookup_options(element, 'plot').options
        style_opts = self.lookup_options(element, 'style').kwargs
        return univariate_kde(element, bandwidth=plot_opts.get('bw')).opts(plot=plot_opts, style=style_opts)



class BivariatePlot(PolygonPlot):
    """
    Bivariate plot visualizes two-dimensional kernel density
    estimates. Additionally, by enabling the joint option, the
    marginals distributions can be plotted alongside each axis (does
    not animate or compose).
    """

    bw = param.Number(default=None)

    filled = param.Boolean(default=False)

    def __init__(self, element, plot=None, **params):
        element = element.map(self._convert_element, Bivariate)
        super(BivariatePlot, self).__init__(element, plot, batched=True, **params)

    def _convert_element(self, element):
        plot_opts = self.lookup_options(element, 'plot').options
        style_opts = self.lookup_options(element, 'style').kwargs
        return bivariate_kde(element, contours=True, filled=plot_opts.get('filled', self.filled),
                             bandwidth=plot_opts.get('bw')).opts(plot=plot_opts, style=style_opts)

    def get_data(self, element, ranges, style):
        data, mapping, style = super(BivariatePlot, self).get_data(element, ranges, style)
        if not self.filled and 'fill_color' in mapping:
            mapping['line_color'] = mapping.pop('fill_color')
        if self.filled:
            style['line_color'] = 'black'
        else:
            style['fill_alpha'] = 0
        return data, mapping, style
