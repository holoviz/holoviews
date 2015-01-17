"""
The holoviews seaborn interface wraps around a wide range
of seaborn plot types including time series, kernel density
estimates, distributions and regression plots.
"""

from __future__ import absolute_import

import numpy as np

import param

from ..core import Dimension, ViewMap, Layer, Overlay
from ..core.options import options, StyleOpts, Cycle
from ..view import DataView, Scatter, Curve
from .pandas import DFrame as PandasDFrame


class TimeSeries(Layer):
    """
    TimeSeries is a container for any set of curves, which the
    seaborn interface combines into a confidence interval, error
    bar or overlaid plot.

    The curves should be supplied as an NxM dimensional array,
    x-values may also be supplied and must be of length N or M.

    Alternatively a Map or Overlay of Curve objects may be
    supplied.
    """

    value = param.String(default='TimeSeries')

    value_dimensions = param.List(default=[Dimension('X')], bounds=(1,1))

    value_dimensions = param.List(default=[Dimension('Y'),
                                           Dimension('Observation')],
                                  bounds=(2,2))

    def __init__(self, data, xvals=None, **params):
        if isinstance(data, ViewMap):
            self.xdata = data.values()[0].data[:, 0]
            data = np.array([dv.data[:, 1] for dv in data.values()])
        elif isinstance(data, Overlay):
            self.xdata = data.data.values()[0].data[:, 0]
            data = np.array([dv.data[:, 1] for dv in data])
        else:
            self.xdata = np.array(range(len(data[0, :]))) if xvals is None\
                else xvals
        super(TimeSeries, self).__init__(data, **params)


    def __getitem__(self, slc):
        if slc is ():
            return self
        else:
            raise NotImplementedError('Slicing and indexing of'
                                      'TimeSeries currently not '
                                      'implemented.')


    def sample(self, **samples):
        raise NotImplementedError('Cannot sample a TimeSeries.')


    def reduce(self, **dimreduce_map):
        raise NotImplementedError('Reduction of TimeSeries not '
                                  'implemented.')


    @property
    def xlim(self):
        if self._xlim:
            return self._xlim
        elif self.cyclic_range is not None:
            return (0, self.cyclic_range)
        else:
            x_vals = self.xdata
            return (float(min(x_vals)), float(max(x_vals)))


    @property
    def ylim(self):
        if self._ylim:
            return self._ylim
        else:
            return (np.min(self.data), np.max(self.data))



class Bivariate(DataView):
    """
    Bivariate Views are containers for two dimensional data,
    which is to be visualized as a kernel density estimate. The
    data should be supplied as an Nx2 array, containing the x-
    and y-data.
    """

    index_dimensions = param.List(default=[Dimension('X'), Dimension('Y')])

    value_dimensions = param.List(default=[], bounds=(0,0))

    value = param.ClassSelector(class_=Dimension, default=None)

    @property
    def xlabel(self):
        return str(self.index_dimensions[0])

    @property
    def ylabel(self):
        return str(self.index_dimensions[1])



class Distribution(DataView):
    """
    Distribution Views provide a container for data to be
    visualized as a one-dimensional distribution. The data should
    be supplied as a simple one-dimensional array or
    list. Internally it uses seaborn to make all the conversions.
    """

    index_dimensions = param.List(default=[], bounds=(0,0))

    value = param.String(default='Distribution')

    value_dimensions = param.List(default=[Dimension('Value'), Dimension('Frequency')])

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
            raise KeyError("Dimension %s not found" % str(dimension))


    def __getitem__(self, index):
        raise NotImplementedError


class Regression(Scatter):
    """
    Regression is identical to a Scatter plot but is visualized
    using the seaborn regplot interface. This allows it to
    implement linear regressions, confidence intervals and a lot
    more.
    """

    value = param.String(default='Regression')


class DFrame(PandasDFrame):
    """
    The SNSFrame is largely the same as a DFrame but can only be
    visualized via seaborn plotting functions. Since most seaborn
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
                                              None],
                                     doc="""Selects which Pandas or Seaborn plot
                                            type to use, when visualizing the plot.""")

    def bivariate(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=Bivariate, **kwargs))

    def distribution(self, value_dim, map_dims=[]):
        if map_dims:
            map_groups = self.data.groupby(map_dims)
            vm_dims = map_dims
        else:
            map_groups = [(0, self.data)]
            vm_dims = ['None']

        vmap = ViewMap(index_dimensions=vm_dims)
        for map_key, group in map_groups:
            vmap[map_key] = Distribution(np.array(group[value_dim]),
                                         index_dimensions=[self.get_dimension(value_dim)])
        return vmap if map_dims else vmap.last

    def regression(self, *args, **kwargs):
        return self.table(*args, **dict(view_type=Regression, **kwargs))

    def timeseries(self, value_dim, dimensions, ts_dims, reduce_fn=None, map_dims=[], **kwargs):
        curve_map = self.table(value_dim, dimensions, reduce_fn=reduce_fn,
                               map_dims=ts_dims+map_dims, **dict(view_type=Curve, **kwargs))
        return TimeSeries(curve_map.overlay(ts_dims))

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



options.TimeSeries = StyleOpts(color=Cycle())
options.Bivariate = StyleOpts(cmap=Cycle(['Blues', 'Oranges', 'PuBu']))
options.Distribution = StyleOpts(color=Cycle())
options.Regression = StyleOpts(color=Cycle())
