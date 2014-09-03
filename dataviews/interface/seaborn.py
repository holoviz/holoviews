"""
The DataViews seaborn interface wraps around a wide range
of seaborn plot types including time series, kernel density
estimates, distributions and regression plots.
"""

from __future__ import absolute_import

import numpy as np

import param

from ..ndmapping import Dimension
from ..views import Overlay
from ..dataviews import DataLayer, Scatter, DataStack
from ..options import options, StyleOpts, Cycle
from .pandas import DFrame as PandasDFrame


class TimeSeries(DataLayer):
    """
    TimeSeries is a container for any set of curves, which the
    seaborn interface combines into a confidence interval, error
    bar or overlaid plot.

    The curves should be supplied as an NxM dimensional array,
    x-values may also be supplied and must be of length N or M.

    Alternatively a Stack or Overlay of Curve objects may be
    supplied.
    """

    def __init__(self, data, xvals=None, **params):
        if isinstance(data, DataStack):
            self.xdata = data.values()[0].data[:,0]
            data = np.array([dv.data[:, 1] for dv in data.values()])
        elif isinstance(data, Overlay):
            self.xdata = data.data[0].data[:,0]
            data = np.array([dv.data[:, 1] for dv in data.data])
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
        return (np.min(self.data), np.max(self.data))



class Bivariate(Scatter):
    """
    Bivariate Views are containers for two dimensional data,
    which is to be visualized as a kernel density estimate. The
    data should be supplied as an Nx2 array, containing the x-
    and y-data.
    """

    @property
    def xlabel(self):
        return str(self.dimensions[0])

    @property
    def ylabel(self):
        return str(self.dimensions[1])



class Distribution(DataLayer):
    """
    Distribution Views provide a container for data to be
    visualized as a one-dimensional distribution. The data should
    be supplied as a simple one-dimensional array or
    list. Internally it uses seaborn to make all the conversions.
    """

    value = param.ClassSelector(class_=(str, Dimension),
                                default='Frequency')

    @property
    def xlim(self):
        if self._xlim:
            return self._xlim
        elif isinstance(self, Overlay):
            return None
        elif self.cyclic_range is not None:
            return (0, self.cyclic_range)
        else:
            return (np.min(self.data), np.max(self.data))

    @property
    def ylim(self):
        return (np.NaN, np.NaN)


    def __getitem__(self, index):
        raise NotImplementedError


class Regression(Scatter):
    """
    Regression is identical to a Scatter plot but is visualized
    using the seaborn regplot interface. This allows it to
    implement linear regressions, confidence intervals and a lot
    more.
    """


class DFrame(PandasDFrame):
    """
    The SNSFrame is largely the same as a DFrame but can only be
    visualized via seaborn plotting functions. Since most seaborn
    plots are two dimensional, the x and y dimensions can be set
    directly on this class to visualize a particular relationship
    in a multi-dimensional Pandas dframe.
    """

    plot_type = param.ObjectSelector(default=None, objects=['interact', 'regplot',
                                                            'lmplot', 'corrplot',
                                                            'plot', 'boxplot',
                                                            'hist', 'scatter_matrix',
                                                            'autocorrelation_plot',
                                                            None],
                                     doc="""Selects which Pandas or Seaborn plot
                                            type to use, when visualizing the plot.""")

    x2 = param.String(doc="""Dimension to visualize along a second
                             dependent axis.""")


options.TimeSeries = StyleOpts(color=Cycle())
options.Bivariate = StyleOpts(cmap=Cycle(['Blues', 'Oranges', 'PuBu']))
options.Distribution = StyleOpts(color=Cycle())
options.Regression = StyleOpts(color=Cycle())
options.SNSFrame = StyleOpts()
