"""
The HoloViews Seaborn interface wraps around a wide range
of Seaborn plot types including time series, kernel density
estimates, distributions and regression plots.
"""

from __future__ import absolute_import

import numpy as np

import param

from ..core import Dimension, NdMapping, Element2D, HoloMap
from ..element import Chart, Scatter, Curve
from .pandas import DFrame as PandasDFrame


class TimeSeries(Element2D):
    """
    TimeSeries is a container for any set of curves, which the
    Seaborn interface combines into a confidence interval, error
    bar or overlaid plot.

    The curves should be supplied as an NxM dimensional array,
    x-values may also be supplied and must be of length N or M.

    Alternatively a UniformNdMapping or NdOverlay of Curve objects may be
    supplied.
    """

    key_dimensions = param.List(default=[Dimension('x'), Dimension('n')],
                                bounds=(2, 2))

    group = param.String(default='TimeSeries')

    value_dimensions = param.List(default=[Dimension('z')],
                                  bounds=(1, 1))

    def __init__(self, data, xdata=None, **params):
        if isinstance(data, NdMapping):
            self.xdata = data.values()[0].data[:, 0]
            params = dict(data.values()[0].get_param_values(onlychanged=True), **params)
            data = np.array([dv.data[:, 1] for dv in data])
        else:
            self.xdata = np.array(range(len(data[0, :]))) if xdata is None\
                else xdata
        super(TimeSeries, self).__init__(data, **params)


    def dimension_values(self, dimension):
        dim_idx = self.get_dimension_index(dimension)
        if dim_idx == 0:
            return self.xdata
        elif dim_idx == 1:
            return self.data.flatten()
        elif dim_idx == 2:
            return range(self.data.shape[1])
        else:
            return super(TimeSeries, self).dimension_values(dimension)


    def sample(self, **samples):
        raise NotImplementedError('Cannot sample a TimeSeries.')


    def reduce(self, **dimreduce_map):
        raise NotImplementedError('Reduction of TimeSeries not '
                                  'implemented.')

    @property
    def ylabel(self):
        return str(self.value_dimensions[0])



class Bivariate(Chart):
    """
    Bivariate Views are containers for two dimensional data,
    which is to be visualized as a kernel density estimate. The
    data should be supplied as an Nx2 array, containing the x-
    and y-data.
    """

    key_dimensions = param.List(default=[Dimension('x'), Dimension('y')])

    value_dimensions = param.List(default=[], bounds=(0,0))

    group = param.String(default="Bivariate")



class Distribution(Chart):
    """
    Distribution Views provide a container for data to be
    visualized as a one-dimensional distribution. The data should
    be supplied as a simple one-dimensional array or
    list. Internally it uses Seaborn to make all the conversions.
    """

    key_dimensions = param.List(default=[Dimension('Value')], bounds=(1,1))

    group = param.String(default='Distribution')

    value_dimensions = param.List(default=[Dimension('Frequency')])

    @property
    def xlim(self):
        return self.range(0)

    @property
    def ylim(self):
        return (None, None)

    def dimension_values(self, dimension):
        dim_idx = self.get_dimension_index(dimension)
        if dim_idx == 0:
            return self.data
        elif dim_idx == 1:
            return []
        else:
            return super(Distribution, self).dimension_values(dimension)


class Regression(Scatter):
    """
    Regression is identical to a Scatter plot but is visualized
    using the Seaborn regplot interface. This allows it to
    implement linear regressions, confidence intervals and a lot
    more.
    """

    group = param.String(default='Regression')


class DFrame(PandasDFrame):
    """
    The SNSFrame is largely the same as a DFrame but can only be
    visualized via Seaborn plotting functions. Since most Seaborn
    plots are two dimensional, the x and y dimensions can be set
    directly on this class to visualize a particular relationship
    in a multi-dimensional Pandas dframe.
    """

    plot_type = param.ObjectSelector(default=None,
                                     objects=['interact', 'regplot',
                                              'lmplot', 'corrplot',
                                              'plot', 'boxplot',
                                              'hist', 'scatter_matrix',
                                              'autocorrelation_plot',
                                              'pairgrid', 'facetgrid',
                                              'pairplot', 'violinplot',
                                              'factorplot',
                                              None],
                                     doc="""Selects which Pandas or Seaborn plot
                                            type to use, when visualizing the plot.""")

    def bivariate(self, kdims, vdims, **kwargs):
        return self.table(kdims, vdims **dict(view_type=Bivariate, **kwargs))

    def distribution(self, value_dim):
        selected_dims = [value_dim]
        map_dims = [dim for dim in self.dimensions(label=True) if dim not in selected_dims]
        if map_dims:
            map_groups = self.data.groupby(map_dims)
            vm_dims = map_dims
        else:
            map_groups = [(0, self.data)]
            vm_dims = ['None']

        vmap = HoloMap(key_dimensions=vm_dims)
        for map_key, group in map_groups:
            vmap[map_key] = Distribution(np.array(group[value_dim]),
                                         key_dimensions=[self.get_dimension(value_dim)])
        return vmap if map_dims else vmap.last

    def regression(self, kdims, vdims, **kwargs):
        return self.table(kdims, vdims, **dict(view_type=Regression, **kwargs))

    def timeseries(self, kdims, vdims, **kwargs):
        if not isinstance(kdims, list) or not len(kdims) ==2:
            raise Exception('TimeSeries requires two key_dimensions.')
        curve_map = self.table(kdims[0], vdims, **dict(view_type=Curve, **kwargs))
        return TimeSeries(curve_map.overlay(kdims[1]),
                          key_dimensions=[self.get_dimension(dim) for dim in kdims],
                          **kwargs)

    @property
    def ylabel(self):
        return self.x2 if self.x2 else self.y

    @property
    def ylim(self):
        if self._ylim:
            return self._ylim
        elif self.x2 or self.y:
            ydata = self.data[self.x2 if self.x2 else self.y]
            return min(ydata), max(ydata)
        else:
            return None
