import numpy as np

import param

from ..core import util
from ..core import Dimension, Dataset, Element2D
from ..core.data import GridInterface
from .geom import Points, VectorField # noqa: backward compatible import
from .stats import BoxWhisker         # noqa: backward compatible import


class Chart(Dataset, Element2D):
    """
    A Chart is an abstract baseclass for elements representing one or
    more independent and dependent variables defining a 1D coordinate
    system with associated values. The independent variables or key
    dimensions map onto the x-axis while the dependent variables are
    usually mapped to the location, height or spread along the
    y-axis. Any number of additional value dimensions may be
    associated with a Chart.

    If a chart's independent variable (or key dimension) is numeric
    the chart will represent a discretely sampled version of the
    underlying continuously sampled 1D space. Therefore indexing along
    this variable will automatically snap to the closest coordinate.

    Since a Chart is a subclass of a Dataset it supports the full set
    of data interfaces but usually each dimension of a chart represents
    a column stored in a dictionary, array or DataFrame.
    """

    kdims = param.List(default=[Dimension('x')], bounds=(1,2), doc="""
        The key dimension(s) of a Chart represent the independent
        variable(s).""")

    group = param.String(default='Chart', constant=True)

    vdims = param.List(default=[Dimension('y')], bounds=(1, None), doc="""
        The value dimensions of the Chart, usually corresponding to a
        number of dependent variables.""")

    # Enables adding index if 1D array like data is supplied
    _auto_indexable_1d = True

    __abstract = True

    def __getitem__(self, index):
        return super(Chart, self).__getitem__(index)


class Scatter(Chart):
    """
    Scatter is a Chart element representing a set of points in a 1D
    coordinate system where the key dimension maps to the points
    location along the x-axis while the first value dimension
    represents the location of the point along the y-axis.
    """

    group = param.String(default='Scatter', constant=True)



class Curve(Chart):
    """
    Curve is a Chart element representing a line in a 1D coordinate
    system where the key dimension maps on the line x-coordinate and
    the first value dimension represents the height of the line along
    the y-axis.
    """

    group = param.String(default='Curve', constant=True)



class ErrorBars(Chart):
    """
    ErrorBars is a Chart element representing error bars in a 1D
    coordinate system where the key dimension corresponds to the
    location along the x-axis and the value dimensions define the
    location along the y-axis and the symmetric or assymetric spread.
    """
    group = param.String(default='ErrorBars', constant=True, doc="""
        A string describing the quantity measured by the ErrorBars
        object.""")

    vdims = param.List(default=[Dimension('y'), Dimension('yerror')],
                       bounds=(1, None), constant=True)


    def range(self, dim, data_range=True, dimension_range=True):
        """Return the lower and upper bounds of values along dimension.

        Range of the y-dimension includes the symmetric or assymetric
        error.

        Args:
            dimension: The dimension to compute the range on.
            data_range (bool): Compute range from data values
            dimension_range (bool): Include Dimension ranges
                Whether to include Dimension range and soft_range
                in range calculation

        Returns:
            Tuple containing the lower and upper bound
        """
        didx = self.get_dimension_index(dim)
        dim = self.get_dimension(dim)
        if didx == 1 and data_range and len(self):
            mean = self.dimension_values(1)
            neg_error = self.dimension_values(2)
            if len(self.dimensions()) > 3:
                pos_error = self.dimension_values(3)
            else:
                pos_error = neg_error
            lower = np.nanmin(mean-neg_error)
            upper = np.nanmax(mean+pos_error)
            if not dimension_range:
                return (lower, upper)
            return util.dimension_range(lower, upper, dim.range, dim.soft_range)
        return super(ErrorBars, self).range(dim, data_range)



class Spread(ErrorBars):
    """
    Spread is a Chart element representing a spread of values or
    confidence band in a 1D coordinate system. The key dimension(s)
    corresponds to the location along the x-axis and the value
    dimensions define the location along the y-axis as well as the
    symmetric or assymetric spread.
    """

    group = param.String(default='Spread', constant=True)



class Bars(Chart):
    """
    Bars is a Chart element representing categorical observations
    using the height of rectangular bars. The key dimensions represent
    the categorical groupings of the data, but may also be used to
    stack the bars, while the first value dimension represents the
    height of each bar.
    """

    group = param.String(default='Bars', constant=True)

    kdims = param.List(default=[Dimension('x')], bounds=(1,3))



class Histogram(Chart):
    """
    Histogram is a Chart element representing a number of bins in a 1D
    coordinate system. The key dimension represents the binned values,
    which may be declared as bin edges or bin centers, while the value
    dimensions usually defines a count, frequency or density associated
    with each bin.
    """

    datatype = param.List(default=['grid'])

    group = param.String(default='Histogram', constant=True)

    kdims = param.List(default=[Dimension('x')], bounds=(1,1), doc="""
        Dimensions on Element2Ds determine the number of indexable
        dimensions.""")

    vdims = param.List(default=[Dimension('Frequency')], bounds=(1, None))

    _binned = True

    def __init__(self, data, edges=None, **params):
        if data is None:
            data = []
        if edges is not None:
            self.param.warning(
                "Histogram edges should be supplied as a tuple "
                "along with the values, passing the edges will "
                "be deprecated in holoviews 2.0.")
            data = (edges, data)
        elif isinstance(data, tuple) and len(data) == 2 and len(data[0])+1 == len(data[1]):
            data = data[::-1]
        super(Histogram, self).__init__(data, **params)


    def __setstate__(self, state):
        """
        Ensures old-style Histogram types without an interface can be unpickled.

        Note: Deprecate as part of 2.0
        """
        if 'interface' not in state:
            self.interface = GridInterface
            x, y = state['_kdims_param_value'][0], state['_vdims_param_value'][0]
            state['data'] = {x.name: state['data'][1], y.name: state['data'][0]}
        super(Dataset, self).__setstate__(state)


    @property
    def values(self):
        "Property to access the Histogram values provided for backward compatibility"
        if util.config.future_deprecations:
            self.param.warning('Histogram.values is deprecated in favor of '
                               'common dimension_values method.')
        return self.dimension_values(1)


    @property
    def edges(self):
        "Property to access the Histogram edges provided for backward compatibility"
        if util.config.future_deprecations:
            self.param.warning('Histogram.edges is deprecated in favor of '
                               'common dimension_values method.')
        return self.interface.coords(self, self.kdims[0], edges=True)


class Spikes(Chart):
    """
    Spikes is a Chart element which represents a number of discrete
    spikes, events or observations in a 1D coordinate system. The key
    dimension therefore represents the position of each spike along
    the x-axis while the first value dimension, if defined, controls
    the height along the y-axis. It may therefore be used to visualize
    the distribution of discrete events, representing a rug plot, or
    to draw the strength some signal.
    """

    group = param.String(default='Spikes', constant=True)

    kdims = param.List(default=[Dimension('x')], bounds=(1, 1))

    vdims = param.List(default=[])

    _auto_indexable_1d = False


class Area(Curve):
    """
    Area is a Chart element representing the area under a curve or
    between two curves in a 1D coordinate system. The key dimension
    represents the location of each coordinate along the x-axis, while
    the value dimension(s) represent the height of the area or the
    lower and upper bounds of the area between curves.

    Multiple areas may be stacked by overlaying them an passing them
    to the stack method.
    """

    group = param.String(default='Area', constant=True)

    @classmethod
    def stack(cls, areas):
        """
        Stacks an (Nd)Overlay of Area or Curve Elements by offsetting
        their baselines. To stack a HoloMap or DynamicMap use the map
        method.
        """
        if not len(areas):
            return areas
        baseline = np.zeros(len(areas.values()[0]))
        stacked = areas.clone(shared_data=False)
        vdims = [areas.values()[0].vdims[0], 'Baseline']
        for k, area in areas.items():
            x, y = (area.dimension_values(i) for i in range(2))
            stacked[k] = area.clone((x, y+baseline, baseline), vdims=vdims,
                                    new_type=Area)
            baseline = baseline + y
        return stacked
