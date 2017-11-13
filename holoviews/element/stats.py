import param
import numpy as np

from ..core.dimension import Dimension, process_dimensions
from ..core.element import Element
from ..core.util import get_param_values, OrderedDict
from .chart import Chart


class StatisticsElement(Chart):
    """
    StatisticsElement provides a baseclass for Element types that
    compute statistics based on the input data. The baseclass
    overrides standard Dataset methods emulating the existence
    of the value dimensions.
    """

    __abstract = True

    def __init__(self, data, kdims=None, vdims=None, **params):
        if isinstance(data, Element):
            params.update(get_param_values(data))
            kdims = kdims or data.dimensions()[:len(self.kdims)]
            data = tuple(data.dimension_values(d) for d in kdims)
        params.update(dict(kdims=kdims, vdims=[], _validate_vdims=False))
        super(StatisticsElement, self).__init__(data, **params)
        if not vdims:
            self.vdims = [Dimension('Density')]
        elif len(vdims) > 1:
            raise ValueError("%s expects at most one vdim." %
                             type(self).__name__)
        else:
            self.vdims = process_dimensions(None, vdims)['vdims']


    def range(self, dim, data_range=True):
        iskdim = self.get_dimension(dim) not in self.vdims
        return super(StatisticsElement, self).range(dim, data_range=iskdim)


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
            dimensions = [self.get_dimension(d, strict=True) for d in dimensions
                          if d in dimensions.kdims]
        else:
            dimensions = dimensions.kdims
        return self.interface.dframe(self, dimensions)


    def columns(self, dimensions=None):
        if dimensions is None:
            dimensions = self.kdims
        else:
            dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        return OrderedDict([(d.name, self.dimension_values(d))
                            for d in dimensions if d in self.kdims])



class Bivariate(StatisticsElement):
    """
    Bivariate elements are containers for two dimensional data,
    which is to be visualized as a kernel density estimate. The
    data should be supplied in a tabular format of x- and y-columns.
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2))

    vdims = param.List(default=[Dimension('Density')], bounds=(0,1))

    group = param.String(default="Bivariate", constant=True)



class Distribution(StatisticsElement):
    """
    Distribution elements provides a representation for a
    one-dimensional distribution which can be visualized as a kernel
    density estimate. The data should be supplied in a tabular format
    and will use the first column.
    """

    kdims = param.List(default=[Dimension('Value')], bounds=(1, 1))

    group = param.String(default='Distribution', constant=True)

    vdims = param.List(default=[Dimension('Density')], bounds=(0, 1))

    # Ensure Interface does not add an index
    _auto_indexable_1d = False

