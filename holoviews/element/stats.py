import param
import numpy as np

from ..core.dimension import Dimension, Dimensioned
from .chart import Chart, Scatter


class _StatisticsElement(Chart):
    """
    StatisticsElement provides a baseclass for Element types that
    compute statistics based on the input data. The baseclass
    overrides standard Dataset methods emulating the existence
    of the value dimensions.
    """

    def __init__(self, data, kdims=None, vdims=None, **params):
        super(_StatisticsElement, self).__init__(data, kdims, vdims, **params)
        if not self.vdims:
            self.vdims = [Dimension('Density')]


    def range(self, dim, data_range=True):
        dim = self.get_dimension(dim)
        if dim in self.vdims:
            return Dimensioned.range(self, dim, data_range=True)
        return super(_StatisticsElement, self).range(dim, data_range)


    def dimension_values(self, dim, expanded=True, flat=True):
        """
        Returns the values along a particular dimension. If unique
        values are requested will return only unique values.
        """
        dim = self.get_dimension(dim, strict=True)
        if dim in self.vdims:
            return np.full(len(self), np.NaN)
        return self.interface.values(self, dim, expanded, flat)


    def get_dimension_type(self, dim):
        """
        Returns the specified Dimension type if specified or
        if the dimension_values types are consistent otherwise
        None is returned.
        """
        dim = self.get_dimension(dim)
        if dim is None:
            return None
        elif dim.type is not None:
            return dim.type
        elif dim in self.vdims:
            return np.float64
        return self.interface.dimension_type(self, dim)


    def dframe(self, dimensions=None):
        """
        Returns the data in the form of a DataFrame. Supplying a list
        of dimensions filters the dataframe. If the data is already
        a DataFrame a copy is returned.
        """
        if dimensions:
            dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        else:
            dimensions = dimensions.kdims
        dim = [dim.name for dim in dims if dim in dimensions.kdims]
        return self.interface.dframe(self, dimensions)


    def columns(self, dimensions=None):
        if dimensions is None:
            dimensions = self.kdims
        else:
            dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        return OrderedDict([(d.name, self.dimension_values(d))
                            for d in dimensions if d in self.kdims])



class Bivariate(_StatisticsElement):
    """
    Bivariate Views are containers for two dimensional data,
    which is to be visualized as a kernel density estimate. The
    data should be supplied as an Nx2 array, containing the x-
    and y-data.
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y')])

    vdims = param.List(default=[Dimension('Density')], bounds=(1,1))

    group = param.String(default="Bivariate", constant=True)



class Distribution(_StatisticsElement):
    """
    Distribution Views provide a container for data to be
    visualized as a one-dimensional distribution. The data should
    be supplied as a simple one-dimensional array or
    list. Internally it uses Seaborn to make all the conversions.
    """

    kdims = param.List(default=[Dimension('Value')])

    group = param.String(default='Distribution', constant=True)

    vdims = param.List(default=[Dimension('Density')])

    _auto_indexable_1d = False


class Regression(Scatter):
    """
    Regression is identical to a Scatter plot but is visualized
    using the Seaborn regplot interface. This allows it to
    implement linear regressions, confidence intervals and a lot
    more.
    """

    group = param.String(default='Regression', constant=True)


