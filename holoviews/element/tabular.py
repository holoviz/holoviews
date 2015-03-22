import numpy as np

import param

from ..core import OrderedDict, Dimension, Element, NdElement, HoloMap


class ItemTable(Element):
    """
    A tabular element type to allow convenient visualization of either
    a standard Python dictionary, an OrderedDict or a list of tuples
    (i.e. input suitable for an OrderedDict constructor). If an
    OrderedDict is used, the headings will be kept in the correct
    order. Tables store heterogeneous data with different labels.

    Dimension objects are also accepted as keys, allowing dimensional
    information (e.g type and units) to be associated per heading.
    """

    key_dimensions = param.List(default=[], bounds=(0, 0), doc="""
       ItemTables hold an index Dimension for each value they contain, i.e.
       they are equivalent to the keys.""")

    value_dimensions = param.List(default=[Dimension('Default')], bounds=(1, None), doc="""
       ItemTables should have only index Dimensions.""")

    group = param.String(default="ItemTable")


    @property
    def rows(self):
        return len(self.value_dimensions)


    @property
    def cols(self):
        return 2


    def __init__(self, data, **params):
        if type(data) == dict:
            raise ValueError("ItemTable cannot accept a standard Python  dictionary "
                             "as a well-defined item ordering is required.")
        elif isinstance(data, dict): pass
        elif isinstance(data, list):
            data = OrderedDict(data)
        else:
            data = OrderedDict(list(data)) # Python 3
        if not 'value_dimensions' in params:
            params['value_dimensions'] = list(data.keys())
        str_keys = OrderedDict((k.name if isinstance(k, Dimension)
                                else k ,v) for (k,v) in data.items())
        super(ItemTable, self).__init__(str_keys, **params)


    def __getitem__(self, heading):
        """
        Get the value associated with the given heading (key).
        """
        if heading is ():
            return self
        if heading not in self._cached_value_names:
            raise IndexError("%r not in available headings." % heading)
        return self.data.get(heading, np.NaN)


    @classmethod
    def collapse_data(cls, data, function, **kwargs):
        if not function:
            raise Exception("Must provide function to collapse %s data." % cls.__name__)
        groups = np.vstack([np.array(odict.values()) for odict in data]).T
        return OrderedDict(zip(data[0].keys(), function(groups, axis=-1, **kwargs)))


    def dimension_values(self, dimension):
        if isinstance(dimension, int):
            dimension = self._cached_index_names[dimension]
        elif dimension in self.dimensions('value', label=True):
            return [self.data.get(dimension, np.NaN)]
        else:
            return super(ItemTable, self).dimension_values(dimension)


    def sample(self, samples=[]):
        if callable(samples):
            sampled_data = OrderedDict(item for item in self.data.items()
                                       if samples(item))
        else:
            sampled_data = OrderedDict((s, self.data.get(s, np.NaN)) for s in samples)
        return self.clone(sampled_data)


    def reduce(self, dimensions=None, function=None, **reduce_map):
        raise NotImplementedError('ItemTables are for heterogeneous data, which'
                                  'cannot be reduced.')


    def pprint_cell(self, row, col):
        """
        Get the formatted cell value for the given row and column indices.
        """
        if col > 2:
            raise Exception("Only two columns available in a ItemTable.")
        elif row >= self.rows:
            raise Exception("Maximum row index is %d" % self.rows-1)
        elif col == 0:
            return str(self.dimensions('value')[row])
        else:
            dim = self.get_dimension(row)
            heading = self._cached_value_names[row]
            return dim.pprint_value(self.data.get(heading, np.NaN))


    def hist(self, *args, **kwargs):
        raise NotImplementedError("ItemTables are not homogenous and "
                                  "don't support histograms.")


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        if col == 0:  return 'heading'
        else:         return 'data'


    def dframe(self):
        """
        Generates a Pandas dframe from the ItemTable.
        """
        from pandas import DataFrame
        return DataFrame({(k.name if isinstance(k, Dimension)
                           else k): [v] for k, v in self.data.items()})


    def table(self):
        return Table(OrderedDict([((), self.values())]), key_dimensions=[],
                     value_dimensions=self.value_dimensions)

    def values(self):
        return tuple(self.data.get(k, np.NaN)
                     for k in self._cached_value_names)



class Table(NdElement):
    """
    Table is an NdElement type, which gets displayed in a tabular
    format and is convertible to most other Element types.
    """

    group = param.String(default='Table', doc="""
         The group is used to describe the Table.""")

    def __setitem__(self, key, value):
        if isinstance(value, (dict, OrderedDict)):
            if all(isinstance(k, str) for k in key):
                value = ItemTable(value)
            else:
                raise ValueError("Tables only supports string inner"
                                 "keys when supplied nested dictionary")
        if isinstance(value, ItemTable):
            if value.value_dimensions != self.value_dimensions:
                raise Exception("Input ItemTables dimensions must match value dimensions.")
            value = value.data.values()
        super(Table, self).__setitem__(key, value)

    @property
    def rows(self):
        return len(self.data) + 1

    @property
    def cols(self):
        return self.ndims + len(self.value_dimensions)


    def pprint_cell(self, row, col):
        """
        Get the formatted cell value for the given row and column indices.
        """
        ndims = self.ndims
        if col >= self.cols:
            raise Exception("Maximum column index is %d" % self.cols-1)
        elif row >= self.rows:
            raise Exception("Maximum row index is %d" % self.rows-1)
        elif row == 0:
            if col >= ndims:
                return str(self.value_dimensions[col - ndims])
            return str(self.key_dimensions[col])
        else:
            dim = self.get_dimension(col)
            if col >= ndims:
                row_values = self.values()[row-1]
                val = row_values[col - ndims]
            else:
                row_data = list(self.data.keys())[row-1]
                val = row_data[col]
            return dim.pprint_value(val)


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        return 'heading' if row == 0 else 'data'


    @property
    def to(self):
        """
        Property to create a conversion table with methods to convert
        to any type.
        """
        return TableConversion(self)



class TableConversion(object):
    """
    TableConversion is a very simple container object which can
    be given an existing Table and provides methods to convert
    the Table into most other Element types.
    """

    def __init__(self, table):
        self._table = table

    def _conversion(self, key_dimensions=None, value_dimensions=None, new_type=None, **kwargs):
        if key_dimensions is None:
            key_dimensions = self._table._cached_index_names
        elif key_dimensions and not isinstance(key_dimensions, list): key_dimensions = [key_dimensions]
        if value_dimensions is None:
            value_dimensions = self._table._cached_value_names
        elif value_dimensions and not isinstance(value_dimensions, list): value_dimensions = [value_dimensions]
        all_dims = self._table.dimensions(label=True)
        invalid = [dim for dim in key_dimensions+value_dimensions if dim not in all_dims]
        if invalid:
            raise Exception("Dimensions %r could not be found during conversion to %s new_type" %
                            (invalid, new_type.__name__))
        group_dims = [dim for dim in self._table._cached_index_names if not dim in key_dimensions]
        selected = self._table.select(**{'value': value_dimensions})
        params = dict({'key_dimensions': [self._table.get_dimension(kd) for kd in key_dimensions],
                       'value_dimensions': [self._table.get_dimension(vd) for vd in value_dimensions]},
                       **kwargs)
        if len(key_dimensions) == self._table.ndims:
            return new_type(selected, **params)
        return selected.groupby(group_dims, container_type=HoloMap, group_type=new_type, **params)

    def bars(self, key_dimensions, value_dimensions, **kwargs):
        from .chart import Bars
        return self._conversion(key_dimensions, value_dimensions, Bars, **kwargs)

    def curve(self, key_dimensions, value_dimensions, **kwargs):
        from .chart import Curve
        return self._conversion(key_dimensions, value_dimensions, Curve, **kwargs)

    def heatmap(self, key_dimensions, value_dimensions, **kwargs):
        from .raster import HeatMap
        return self._conversion(key_dimensions, value_dimensions, HeatMap, **kwargs)

    def points(self, key_dimensions, value_dimensions, **kwargs):
        from .chart import Points
        return self._conversion(key_dimensions, value_dimensions, Points, **kwargs)

    def scatter(self, key_dimensions, value_dimensions, **kwargs):
        from .chart import Scatter
        return self._conversion(key_dimensions, value_dimensions, Scatter, **kwargs)

    def scatter3d(self, key_dimensions, value_dimensions, **kwargs):
        from .chart3d import Scatter3D
        return self._conversion(key_dimensions, value_dimensions, Scatter3D, **kwargs)

    def surface(self, key_dimensions, value_dimensions, **kwargs):
        from .chart3d import Surface
        heatmap = self.to_heatmap(key_dimensions, value_dimensions, **kwargs)
        return Surface(heatmap.data, **dict(self._table.get_param_values(onlychanged=True)))

    def vectorfield(self, key_dimensions, value_dimensions, **kwargs):
        from .chart import VectorField
        return self._conversion(key_dimensions, value_dimensions, VectorField, **kwargs)
