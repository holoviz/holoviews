import param
import numpy as np

from ..core.dimension import Dimension, process_dimensions
from ..core.data import Dataset
from ..core.element import Element, Element2D
from ..core.util import get_param_values, unique_iterator, OrderedDict
from .selection import Selection1DExpr, Selection2DExpr


class StatisticsElement(Dataset, Element2D):
    """
    StatisticsElement provides a baseclass for Element types that
    compute statistics based on the input data, usually a density.
    The value dimension of such elements are therefore usually virtual
    and not computed until the element is plotted.
    """

    __abstract = True

    # Ensure Interface does not add an index
    _auto_indexable_1d = False

    def __init__(self, data, kdims=None, vdims=None, **params):
        if (isinstance(data, Element) and
                data.interface.datatype != "dataframe"):
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

    @property
    def dataset(self):
        """
        The Dataset that this object was created from
        """
        from . import Dataset
        if self._dataset is None:
            datatype = list(unique_iterator(self.datatype+Dataset.datatype))
            dataset = Dataset(self, dataset=None, pipeline=None, transforms=None,
                              vdims=[], datatype=datatype)
            return dataset
        elif not isinstance(self._dataset, Dataset):
            return Dataset(self, _validate_vdims=False, **self._dataset)
        return self._dataset


    def range(self, dim, data_range=True, dimension_range=True):
        """Return the lower and upper bounds of values along dimension.

        Args:
            dimension: The dimension to compute the range on.
            data_range (bool): Compute range from data values
            dimension_range (bool): Include Dimension ranges
                Whether to include Dimension range and soft_range
                in range calculation

        Returns:
            Tuple containing the lower and upper bound
        """
        iskdim = self.get_dimension(dim) not in self.vdims
        return super(StatisticsElement, self).range(dim, iskdim, dimension_range)


    def dimension_values(self, dim, expanded=True, flat=True):
        """Return the values along the requested dimension.

        Args:
            dimension: The dimension to return values for
            expanded (bool, optional): Whether to expand values
                Whether to return the expanded values, behavior depends
                on the type of data:
                  * Columnar: If false returns unique values
                  * Geometry: If false returns scalar values per geometry
                  * Gridded: If false returns 1D coordinates
            flat (bool, optional): Whether to flatten array

        Returns:
            NumPy array of values along the requested dimension
        """
        dim = self.get_dimension(dim, strict=True)
        if dim in self.vdims:
            return np.full(len(self), np.NaN)
        return self.interface.values(self, dim, expanded, flat)


    def get_dimension_type(self, dim):
        """Get the type of the requested dimension.

        Type is determined by Dimension.type attribute or common
        type of the dimension values, otherwise None.

        Args:
            dimension: Dimension to look up by name or by index

        Returns:
            Declared type of values along the dimension
        """
        dim = self.get_dimension(dim)
        if dim is None:
            return None
        elif dim.type is not None:
            return dim.type
        elif dim in self.vdims:
            return np.float64
        return self.interface.dimension_type(self, dim)


    def dframe(self, dimensions=None, multi_index=False):
        """Convert dimension values to DataFrame.

        Returns a pandas dataframe of columns along each dimension,
        either completely flat or indexed by key dimensions.

        Args:
            dimensions: Dimensions to return as columns
            multi_index: Convert key dimensions to (multi-)index

        Returns:
            DataFrame of columns corresponding to each dimension
        """
        if dimensions:
            dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        else:
            dimensions = self.kdims
        vdims = [d for d in dimensions if d in self.vdims]
        if vdims:
            raise ValueError('%s element does not hold data for value '
                             'dimensions. Could not return data for %s '
                             'dimension(s).' %
                             (type(self).__name__, ', '.join([d.name for d in vdims])))
        return super(StatisticsElement, self).dframe(dimensions, False)


    def columns(self, dimensions=None):
        """Convert dimension values to a dictionary.

        Returns a dictionary of column arrays along each dimension
        of the element.

        Args:
            dimensions: Dimensions to return as columns

        Returns:
            Dictionary of arrays for each dimension
        """
        if dimensions is None:
            dimensions = self.kdims
        else:
            dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        vdims = [d for d in dimensions if d in self.vdims]
        if vdims:
            raise ValueError('%s element does not hold data for value '
                             'dimensions. Could not return data for %s '
                             'dimension(s).' %
                             (type(self).__name__, ', '.join([d.name for d in vdims])))
        return OrderedDict([(d.name, self.dimension_values(d)) for d in dimensions])



class Bivariate(Selection2DExpr, StatisticsElement):
    """
    Bivariate elements are containers for two dimensional data, which
    is to be visualized as a kernel density estimate. The data should
    be supplied in a tabular format of x- and y-columns.
    """

    group = param.String(default="Bivariate", constant=True)

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2))

    vdims = param.List(default=[Dimension('Density')], bounds=(0,1))



class Distribution(Selection1DExpr, StatisticsElement):
    """
    Distribution elements provides a representation for a
    one-dimensional distribution which can be visualized as a kernel
    density estimate. The data should be supplied in a tabular format
    and will use the first column.
    """

    group = param.String(default='Distribution', constant=True)

    kdims = param.List(default=[Dimension('Value')], bounds=(1, 1))

    vdims = param.List(default=[Dimension('Density')], bounds=(0, 1))



class BoxWhisker(Selection1DExpr, Dataset, Element2D):
    """
    BoxWhisker represent data as a distributions highlighting the
    median, mean and various percentiles. It may have a single value
    dimension and any number of key dimensions declaring the grouping
    of each violin.
    """

    group = param.String(default='BoxWhisker', constant=True)

    kdims = param.List(default=[], bounds=(0, None))

    vdims = param.List(default=[Dimension('y')], bounds=(1,1))

    _inverted_expr = True


class Violin(BoxWhisker):
    """
    Violin elements represent data as 1D distributions visualized
    as a kernel-density estimate. It may have a single value dimension
    and any number of key dimensions declaring the grouping of each
    violin.
    """

    group = param.String(default='Violin', constant=True)


class HexTiles(Selection2DExpr, Dataset, Element2D):
    """
    HexTiles is a statistical element with a visual representation
    that renders a density map of the data values as a hexagonal grid.

    Before display the data is aggregated either by counting the values
    in each hexagonal bin or by computing aggregates.
    """

    group = param.String(default='HexTiles', constant=True)

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2))


