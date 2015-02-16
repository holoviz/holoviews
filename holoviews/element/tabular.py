import numpy as np

import param

from ..core import OrderedDict, Dimension, NdMapping, Element, HoloMap


class ItemTable(Element):
    """
    A tabular element type to allow convenient visualization of either a
    standard Python dictionary or an OrderedDict. If an OrderedDict is
    used, the headings will be kept in the correct order. Tables store
    heterogeneous data with different labels.

    Dimension objects are also accepted as keys, allowing dimensional
    information (e.g type and units) to be associated per heading.
    """

    key_dimensions = param.List(default=[], bounds=(0, 0), doc="""
       ItemTables hold an index Dimension for each value they contain, i.e.
       they are equivalent to the keys.""")

    value_dimensions = param.List(default=[Dimension('Default')], bounds=(1, None), doc="""
       ItemTables should have only index Dimensions.""")

    value = param.String(default="ItemTable")


    @property
    def rows(self):
        return len(self.value_dimensions)


    @property
    def cols(self):
        return 2


    def __init__(self, data, **params):
        # Assume OrderedDict if not a vanilla Python dict
        if type(data) == dict:
            data = OrderedDict(sorted(data.items()))

        str_keys=dict((k.name if isinstance(k, Dimension)
                       else k ,v) for (k,v) in data.items())
        params = dict(params, value_dimensions=list(data.keys()))
        super(ItemTable, self).__init__(str_keys, **params)


    def __getitem__(self, heading):
        """
        Get the value associated with the given heading (key).
        """
        if heading is ():
            return self
        if heading not in self._cached_index_names:
            raise IndexError("%r not in available headings." % heading)
        return self.data[heading]


    @classmethod
    def collapse_data(cls, data, function, **kwargs):
        if not function:
            raise Exception("Must provide function to collapse %s data." % cls.__name__)
        groups = np.vstack([np.array(odict.values()) for odict in data]).T
        return OrderedDict(zip(data[0].keys(), function(groups, axis=-1, **kwargs)))


    def dimension_values(self, dimension):
        if isinstance(dimension, int):
            dimension = self._cached_index_names[dimension]
        return [self.data[dimension]]


    def sample(self, samples=[]):
        if callable(samples):
            sampled_data = OrderedDict(item for item in self.data.items()
                                       if samples(item))
        else:
            sampled_data = OrderedDict((s, self.data[s]) for s in samples)
        return self.clone(sampled_data)


    def reduce(self, dimensions=None, function=None, **reduce_map):
        raise NotImplementedError('ItemTables are for heterogeneous data, which'
                                  'cannot be reduced.')


    def cell_value(self, row, col):
        """
        Get the stored value for a given row and column indices.
        """
        if col > 2:
            raise Exception("Only two columns available in a ItemTable.")
        elif row >= self.rows:
            raise Exception("Maximum row index is %d" % self.rows-1)
        elif col == 0:
            return str(self.dimensions('value')[row])
        else:
            heading = self.dimensions('value', label=True)[row]
            return self.data[heading]


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
        return Table(OrderedDict([((), self.data.values())]),
                     value_dimensions=self.value_dimensions, key_dimensions=[])


class Table(Element, NdMapping):
    """
    A Table is an NdMapping that is rendered in tabular form. In
    addition to the usual multi-dimensional keys of NdMappings
    (rendered as columns), Tables also support multi-dimensional
    values also rendered as columns. The values held in a multi-valued
    Table are tuples, where each component of the tuple maps to a
    column as described by the value_dimensions parameter.

    In other words, the columns of a table are partitioned into two
    groups: the columns based on the key and the value columns that
    contain the components of the value tuple.

    One feature of Tables is that they support an additional level of
    index over NdMappings: the last index may be a column name or a
    slice over the column names (using alphanumeric ordering).
    """

    value = param.String(default='Table', doc="""
         The value Dimension is used to describe the table. Example of
         dimension names include 'Summary' or 'Statistics'. """)

    value_dimensions = param.List(default=[Dimension('Data')],
                                  bounds=(1,None), doc="""
        The dimension description(s) of the values held in data tuples
        that map to the value columns of the table.

        Note: String values may be supplied in the constructor which
        will then be promoted to Dimension objects.""")

    _deep_indexable = False

    def __init__(self, data=None, **params):
        self._style = None
        NdMapping.__init__(self, data, **dict(params,
                                              value=params.get('value',self.value)))
        for k, v in self.data.items():
            self[k] = v # Unpacks any ItemTables


    def __setitem__(self, key, value):
        if isinstance(value, ItemTable):
            indices = []
            if len(value.key_dimensions) != len(self.value_dimensions):
                raise Exception("Input ItemTables dimensions must match value dimensions.")
            for dim in self.value_dimensions:
                idx = [d.name for d in value.key_dimensions].index(dim.name)
                if hash(dim) != hash(value.key_dimensions[idx]):
                    raise Exception("Input ItemTables dimensions must match value dimensions.")
                indices.append(idx)
            value = tuple(value.data.values()[i] for i in indices)
        self.data[key] = value


    def _filter_columns(self, index, col_names):
        "Returns the column names specified by index (which may be a slice)"
        if isinstance(index, slice):
            cols  = [col for col in sorted(col_names)]
            if index.start:
                cols = [col for col in cols if col > index.start]
            if index.stop:
                cols = [col for col in cols if col < index.stop]
            cols = cols[::index.step] if index.step else cols
        elif isinstance(index, list):
            nomatch = [val for val in index if val not in col_names]
            if nomatch:
                raise KeyError("No columns with dimension labels %r" % nomatch)
            cols = [col for col in col_names if col in index]
        elif index not in col_names:
            raise KeyError("No column with dimension label %r" % index)
        else:
            cols= [index]
        if cols==[]:
            raise KeyError("No columns selected in the given slice")
        return cols


    def _filter_table(self, subtable, value_dimensions):
        col_names = self.dimensions('value', label=True)
        cols = self._filter_columns(value_dimensions, col_names)
        indices = [col_names.index(col) for col in cols]
        value_dimensions = [self.value_dimensions[i] for i in indices]
        if isinstance(subtable, ItemTable):
            items = OrderedDict([(h,v) for (h,v) in subtable.data.items() if h in cols])
            return ItemTable(items, label=self.label)

        items = [(k, tuple(v[i] for i in indices)) for (k,v) in subtable.items()]
        return subtable.clone(items, value_dimensions=value_dimensions)


    def __getitem__(self, args):
        """
        In addition to usual NdMapping indexing, Tables can be indexed
        by column name (or a slice over column names)
        """
        ndmap_index = args[:self.ndims] if isinstance(args, tuple) else args
        subtable = NdMapping.__getitem__(self, ndmap_index)

        if len(self.value_dimensions) > 1 and not isinstance(subtable, Table):
            # If a value tuple, turn into an ItemTable
            subtable = ItemTable(OrderedDict(zip(self.value_dimensions, subtable)),
                                 label=self.label)

        if not isinstance(args, tuple) or len(args) <= self.ndims:
            return subtable

        return self._filter_table(subtable, args[-1])


    def select(self, **selection):
        val_selection = None
        if self.value in selection:
            val_selection = selection.pop(self.value)
        selection = NdMapping.select(self, **selection)
        if val_selection:
            return self._filter_table(selection, val_selection)
        else:
            return selection


    @property
    def rows(self):
        return len(self.data) + 1

    @property
    def cols(self):
        return self.ndims + len(self.value_dimensions)


    def cell_value(self, row, col):
        """
        Get the stored value for a given row and column indices.
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
            if col >= ndims:
                row_values = self.values()[row-1]
                return (row_values[col - ndims]
                        if not np.isscalar(row_values) else row_values)
            row_data = list(self.data.keys())[row-1]

            return row_data[col] if isinstance(row_data, tuple) else row_data


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        return 'heading' if row == 0 else 'data'


    def sample(self, samples=[]):
        """
        Allows sampling of the Table with a list of samples.
        """
        sample_data = OrderedDict()
        for sample in samples:
            sample_data[sample] = self[sample]
        return Table(sample_data, **dict(self.get_param_values(onlychanged=True)))


    def reduce(self, dimensions=None, function=None, **reduce_map):
        """
        Allows collapsing the Table down by dimension by passing
        the dimension name and reduce_fn as kwargs. Reduces
        dimensionality of Table until only an ItemTable is left.
        """
        dimensions = self._valid_dimensions(dimensions)
        if dimensions and reduce_map:
            raise Exception("Pass reduced dimensions either as an argument"
                            "or as part of the kwargs not both.")
        elif dimensions:
            reduce_map = {d: function for d in dimensions}

        dim_labels = self._cached_index_names
        reduced_table = self
        for dim, reduce_fn in reduce_map.items():
            split_dims = [self.get_dimension(d) for d in dim_labels if d != dim]
            if len(split_dims) and reduced_table.ndims > 1:
                split_map = reduced_table.groupby([dim])
                reduced_table = self.clone(shared_data=False, key_dimensions=split_dims)
                for k, table in split_map.items():
                    if len(self.value_dimensions) > 1:
                        reduced = tuple(reduce_fn(table.dimension_values(vdim.name))
                                        for vdim in self.value_dimensions)
                    else:
                        reduced = reduce_fn(table.data.values())
                    reduced_table[k] = reduced
            else:
                reduced = {vdim: reduce_fn(self.dimension_values(vdim.name))
                           for vdim in self.value_dimensions}
                params = dict(value=self.value) if self.value != type(self).__name__ else {}
                reduced_table = ItemTable(reduced, label=self.label, **params)
        return reduced_table


    def _item_check(self, dim_vals, data):
        if isinstance(data, tuple):
            for el in data:
                self._item_check(dim_vals, el)
            return
        super(Table, self)._item_check(dim_vals, data)


    @classmethod
    def collapse_data(cls, data, function, **kwargs):
        if not function:
            raise Exception("Must provide function to collapse %s data." % cls.__name__)
        groups = zip(*[(np.array([values]) if np.isscalar(values) else np.array(values)
                        for values in odict.values()) for odict in data])
        return OrderedDict((key, function(np.vstack(group), axis=-1, **kwargs)
                                  if group[0].shape[0] > 1 else
                                  function(np.concatenate(group), **kwargs))
                             for key, group in zip(data[0].keys(), groups))


    def tablemap(self, dimensions):
        if len(dimensions) < self.ndims:
            return self.groupby(dimensions, container_type=HoloMap)
        else:
            vmap = HoloMap(key_dimensions=[self.get_dimension(d) for d in dimensions])
            for k, v in self.items():
                vmap[k] = ItemTable(dict(zip(self.value_dimensions, v)))
            return vmap


    def dimension_values(self, dim):
        if isinstance(dim, Dimension):
            raise Exception('Dimension to be specified by name')
        if dim == self.value:
            return self.values()
        elif dim in self.dimensions('value', label=True):
            if len(self.value_dimensions) == 1: return self.values()
            index = [v.name for v in self.value_dimensions].index(dim)
            return [v[index] for v in self.values()]
        elif dim in self._cached_index_names:
            return NdMapping.dimension_values(self, dim)
        else:
            raise Exception('Dimension not found.')


    @property
    def to(self):
        """
        Property to create a conversion table with methods to convert
        to any type.
        """
        return TableConversion(self)


    def dframe(self, value_label='data'):
        try:
            import pandas
        except ImportError:
            raise Exception("Cannot build a DataFrame without the pandas library.")
        labels = [d.name for d in self.dimensions()]
        return pandas.DataFrame(
            [dict(zip(labels, k+ (v if isinstance(v, tuple) else (v,))))
             for (k, v) in self.data.items()])



class TableConversion(object):
    """
    TableConversion is a very simple container object which can
    be given an existing Table and provides methods to convert
    the Table into most other Element types.
    """

    def __init__(self, table):
        self._table = table

    def _conversion(self, key_dimensions=[], value_dimensions=[], new_type=None):
        if not isinstance(key_dimensions, list): key_dimensions = [key_dimensions]
        if not isinstance(value_dimensions, list): value_dimensions = [value_dimensions]
        all_dims = self._table.dimensions(label=True)
        invalid = [dim for dim in key_dimensions+value_dimensions if dim not in all_dims]
        if invalid:
            raise Exception("Dimensions %r could not be found during conversion to %s new_type" %
                            (invalid, new_type.__name__))
        group_dims = [dim for dim in self._table._cached_index_names if not dim in key_dimensions]
        selected = self._table.select(**{self._table.value: value_dimensions})
        return selected.groupby(group_dims, container_type=HoloMap, group_type=new_type)

    def bars(self, key_dimensions, value_dimensions):
        from .chart import Bars
        return self._conversion(key_dimensions, value_dimensions, Bars)

    def curve(self, key_dimensions, value_dimensions):
        from .chart import Curve
        return self._conversion(key_dimensions, value_dimensions, Curve)

    def heatmap(self, key_dimensions, value_dimensions):
        from .raster import HeatMap
        return self._conversion(key_dimensions, value_dimensions, HeatMap)

    def points(self, key_dimensions, value_dimensions):
        from .chart import Points
        return self._conversion(key_dimensions, value_dimensions, Points)

    def scatter(self, key_dimensions, value_dimensions):
        from .chart import Scatter
        return self._conversion(key_dimensions, value_dimensions, Scatter)

    def scatter3d(self, key_dimensions, value_dimensions):
        from .chart3d import Scatter3D
        return self._conversion(key_dimensions, value_dimensions, Scatter3D)

    def surface(self, key_dimensions, value_dimensions):
        from .chart3d import Surface
        heatmap = self.to_heatmap(key_dimensions, value_dimensions)
        return Surface(heatmap.data, **dict(self._table.get_param_values(onlychanged=True)))

    def vectorfield(self, key_dimensions, value_dimensions):
        from .chart import VectorField
        return self._conversion(key_dimensions, value_dimensions, VectorField)
