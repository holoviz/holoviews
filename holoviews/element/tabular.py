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

    kdims = param.List(default=[], bounds=(0, 0), doc="""
       ItemTables hold an index Dimension for each value they contain, i.e.
       they are equivalent to the keys.""")

    vdims = param.List(default=[Dimension('Default')], bounds=(1, None), doc="""
       ItemTables should have only index Dimensions.""")

    group = param.String(default="ItemTable", constant=True)


    @property
    def rows(self):
        return len(self.vdims)


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
        if not 'vdims' in params:
            params['vdims'] = list(data.keys())
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
        groups = np.vstack([np.array(odict.values()) for odict in data]).T
        return OrderedDict(zip(data[0].keys(), function(groups, axis=-1, **kwargs)))


    def dimension_values(self, dimension):
        dimension = self.get_dimension(dimension).name
        if dimension in self.dimensions('value', label=True):
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
        return Table(OrderedDict([((), self.values())]), kdims=[],
                     vdims=self.vdims)

    def values(self):
        return tuple(self.data.get(k, np.NaN)
                     for k in self._cached_value_names)



class Table(NdElement):
    """
    Table is an NdElement type, which gets displayed in a tabular
    format and is convertible to most other Element types.
    """

    kdims = param.List(default=[Dimension(name="Row")], doc="""
         One or more key dimensions. By default, the special 'Row'
         dimension ensures that the table is always indexed by the row
         number.

         If no key dimensions are set, only one entry can be stored
         using the empty key ().""")

    group = param.String(default='Table', constant=True, doc="""
         The group is used to describe the Table.""")

    def _add_item(self, key, value, sort=True):
        if self.indexed and ((key != len(self)) and (key != (len(self),))):
            raise Exception("Supplied key %s does not correspond to the items row number." % key)

        if isinstance(value, (dict, OrderedDict)):
            if all(isinstance(k, str) for k in key):
                value = ItemTable(value)
            else:
                raise ValueError("Tables only supports string inner"
                                 "keys when supplied nested dictionary")
        if isinstance(value, ItemTable):
            if value.vdims != self.vdims:
                raise Exception("Input ItemTables dimensions must match value dimensions.")
            value = value.data.values()
        super(Table, self)._add_item(key, value, sort)

    @property
    def indexed(self):
        """
        Whether this is an indexed table: a table that has a single
        key dimension called 'Row' corresponds to the row number.
        """
        return self.ndims == 1 and self.kdims[0].name == 'Row'

    @property
    def to(self):
        """
        Property to create a conversion table with methods to convert
        to any type.
        """
        return TableConversion(self)

    def dframe(self, value_label='data'):
        dframe = super(Table, self).dframe(value_label=value_label)
        # Drop 'Row' column as it is redundant with dframe index
        if self.indexed: del dframe['Row']
        return dframe



class TableConversion(object):
    """
    TableConversion is a very simple container object which can
    be given an existing Table and provides methods to convert
    the Table into most other Element types.
    """

    def __init__(self, table):
        self._table = table

    def _conversion(self, kdims=None, vdims=None, new_type=None, **kwargs):
        if kdims is None:
            kdims = self._table._cached_index_names
        elif kdims and not isinstance(kdims, list): kdims = [kdims]
        if vdims is None:
            vdims = self._table._cached_value_names
        elif vdims and not isinstance(vdims, list): vdims = [vdims]
        kdims = [kdim.name if isinstance(kdim, Dimension) else kdim for kdim in kdims]
        vdims = [vdim.name if isinstance(vdim, Dimension) else vdim for vdim in vdims]
        if (any(kd in self._table._cached_value_names for kd in kdims) or
            any(vd in self._table._cached_index_names for vd in vdims)):
            new_kdims = [kd for kd in self._table._cached_index_names
                         if kd not in kdims and kd not in vdims] + kdims
            selected = self._table.reindex(new_kdims, vdims)
        else:
            selected = self._table.select(**{'value': vdims})
        all_dims = selected.dimensions(label=True)
        invalid = [dim for dim in kdims+vdims if dim not in all_dims]
        if invalid:
            raise Exception("Dimensions %r could not be found during conversion to %s new_type" %
                            (invalid, new_type.__name__))
        group_dims = [dim for dim in selected._cached_index_names if not dim in kdims+vdims]

        params = dict({'kdims': [selected.get_dimension(kd) for kd in kdims],
                       'vdims': [selected.get_dimension(vd) for vd in vdims]},
                       **kwargs)
        if len(kdims) == selected.ndims:
            return new_type(selected, **params)
        return selected.groupby(group_dims, container_type=HoloMap, group_type=new_type, **params)

    def bars(self, kdims=None, vdims=None, **kwargs):
        from .chart import Bars
        return self._conversion(kdims, vdims, Bars, **kwargs)

    def curve(self, kdims=None, vdims=None, **kwargs):
        from .chart import Curve
        return self._conversion(kdims, vdims, Curve, **kwargs)

    def heatmap(self, kdims=None, vdims=None, **kwargs):
        from .raster import HeatMap
        return self._conversion(kdims, vdims, HeatMap, **kwargs)

    def points(self, kdims=None, vdims=None, **kwargs):
        from .chart import Points
        return self._conversion(kdims, vdims, Points, **kwargs)

    def scatter(self, kdims=None, vdims=None, **kwargs):
        from .chart import Scatter
        return self._conversion(kdims, vdims, Scatter, **kwargs)

    def scatter3d(self, kdims=None, vdims=None, **kwargs):
        from .chart3d import Scatter3D
        return self._conversion(kdims, vdims, Scatter3D, **kwargs)

    def raster(self, kdims=None, vdims=None, **kwargs):
        from .raster import Raster
        heatmap = self.heatmap(kdims, vdims, **kwargs)
        return Raster(heatmap.data, **dict(self._table.get_param_values(onlychanged=True)))

    def surface(self, kdims=None, vdims=None, **kwargs):
        from .chart3d import Surface
        heatmap = self.heatmap(kdims, vdims, **kwargs)
        return Surface(heatmap.data, **dict(self._table.get_param_values(onlychanged=True)))

    def vectorfield(self, kdims=None, vdims=None, **kwargs):
        from .chart import VectorField
        return self._conversion(kdims, vdims, VectorField, **kwargs)
