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

    kdims = param.List(default=[Dimension('x'), Dimension('n')],
                       bounds=(2, 2))

    group = param.String(default='TimeSeries', constant=True)

    vdims = param.List(default=[Dimension('z')],
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


    def sample(self, samples=[], **sample_values):
        raise NotImplementedError('Cannot sample a TimeSeries.')


    def reduce(self, dimensions=[], function=None, **reduce_map):
        raise NotImplementedError('Reduction of TimeSeries not '
                                  'implemented.')



class Bivariate(Chart):
    """
    Bivariate Views are containers for two dimensional data,
    which is to be visualized as a kernel density estimate. The
    data should be supplied as an Nx2 array, containing the x-
    and y-data.
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y')])

    vdims = param.List(default=[], bounds=(0,1))

    group = param.String(default="Bivariate", constant=True)



class Distribution(Chart):
    """
    Distribution Views provide a container for data to be
    visualized as a one-dimensional distribution. The data should
    be supplied as a simple one-dimensional array or
    list. Internally it uses Seaborn to make all the conversions.
    """

    kdims = param.List(default=[])

    group = param.String(default='Distribution', constant=True)

    vdims = param.List(default=[Dimension('Value')])

    _1d = True


class Regression(Scatter):
    """
    Regression is identical to a Scatter plot but is visualized
    using the Seaborn regplot interface. This allows it to
    implement linear regressions, confidence intervals and a lot
    more.
    """

    group = param.String(default='Regression', constant=True)


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


    def bivariate(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        return self._convert(kdims, vdims, mdims, reduce_fn, view_type=Bivariate, **kwargs)


    def distribution(self, dim, mdims=[], **kwargs):
        grouped = self.groupby(mdims, HoloMap) if mdims else HoloMap({0: self})
        inherited = dict(kdims=[self.get_dimension(dim)],
                         label=self.label)
        kwargs = dict(inherited, **kwargs)
        conversion_fn = lambda x: Distribution(x.data.sort()[dim].dropna(), **kwargs)
        distributions = grouped.map(conversion_fn, [DFrame])
        return distributions if mdims else distributions.last


    def regression(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        return self._convert(kdims, vdims, mdims, reduce_fn, view_type=Regression, **kwargs)


    def timeseries(self, kdims, vdims, mdims=[], reduce_fn=None, **kwargs):
        if not isinstance(kdims, list) or not len(kdims) ==2:
            raise Exception('TimeSeries requires two key dimensions.')
        if not isinstance(kdims, list): kdims = [kdims]
        if not isinstance(vdims, list): vdims = [vdims]

        sel_dims = kdims + vdims
        if mdims:
            mdims = mdims + [kdims[1]]
        else:
            mdims = [kdims[1]]
            if not reduce_fn:
                mdims += [dim for dim in self.dimensions(label=True) if dim not in sel_dims]
        curve_map = self._convert(kdims[0], vdims, mdims, reduce_fn, view_type=Curve, **kwargs)
        return TimeSeries(curve_map.overlay(kdims[1]),
                          kdims=[self.get_dimension(dim) for dim in kdims],
                          **kwargs)


__all__ = ['DFrame', 'Bivariate', 'Distribution',
           'TimeSeries', 'Regression']
