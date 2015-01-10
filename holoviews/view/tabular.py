from collections import OrderedDict

import numpy as np

import param

from ..core import Dimension, Layer, NdMapping, ViewMap


class ItemTable(Layer):
    """
    A tabular view type to allow convenient visualization of either a
    standard Python dictionary or an OrderedDict. If an OrderedDict is
    used, the headings will be kept in the correct order. Tables store
    heterogeneous data with different labels.

    Dimension objects are also accepted as keys, allowing dimensional
    information (e.g type and units) to be associated per heading.
    """

    xlabel, ylabel = None, None
    xlim, ylim = None, None
    lbrt = None, None, None, None

    @property
    def rows(self):
        return self.ndims


    @property
    def cols(self):
        return 2


    def __init__(self, data, **params):
        # Assume OrderedDict if not a vanilla Python dict
        if type(data) == dict:
            data = OrderedDict(sorted(data.items()))

        str_keys=dict((k.name if isinstance(k, Dimension)
                       else k ,v) for (k,v) in data.items())
        params = dict(params, dimensions=data.keys())
        super(ItemTable, self).__init__(str_keys, **params)


    def __getitem__(self, heading):
        """
        Get the value associated with the given heading (key).
        """
        if heading is ():
            return self
        if heading not in self.dim_dict:
            raise IndexError("%r not in available headings." % heading)
        if isinstance(heading, Dimension):
            return self.data[heading.name]
        else:
            return self.data[heading]


    def sample(self, samples=None):
        if callable(samples):
            sampled_data = OrderedDict(item for item in self.data.items()
                                       if samples(item))
        else:
            sampled_data = OrderedDict((s, self.data[s]) for s in samples)
        return self.clone(sampled_data)


    def reduce(self, **reduce_map):
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
            return list(self.dim_dict.values())[row]
        else:
            heading = list(self.dim_dict.keys())[row]
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



class Table(Layer, NdMapping):

    value = param.ClassSelector(class_=Dimension,
                                default=Dimension('Value'), doc="""
        The dimension description of the data held in the data array.""")

    xlabel, ylabel = None, None
    xlim, ylim = None, None
    lbrt = None, None, None, None

    def __init__(self, data=None, **params):
        if 'value' in params and not isinstance(params['value'], Dimension):
            params['value'] = Dimension(params['value'])
        self._style = None
        NdMapping.__init__(self, data, **params)
        self.data = self._data

    def __getitem__(self, *args):
        return NdMapping.__getitem__(self, *args)

    @property
    def range(self):
        values = self.values()
        return (min(values), max(values))

    @property
    def rows(self):
        return len(self._data) + 1

    @property
    def cols(self):
        return self.ndims + 1

    def clone(self, *args, **params):
        return NdMapping.clone(self, *args, **params)


    def cell_value(self, row, col):
        """
        Get the stored value for a given row and column indices.
        """
        if col >= self.cols:
            raise Exception("Maximum column index is %d" % self.cols-1)
        elif row >= self.rows:
            raise Exception("Maximum row index is %d" % self.rows-1)
        elif row == 0:
            if col == self.ndims:
                return str(self.value)
            return str(self.dimensions[col])
        else:
            if col == self.ndims:
                return self.values()[row-1]
            return self._data.keys()[row-1][col]
            heading = list(self.dim_dict.keys())[row]
            return self.data[heading]


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
        return Table(sample_data, **dict(self.get_param_values()))


    def reduce(self, **reduce_map):
        """
        Allows collapsing the Table down by dimension by passing
        the dimension name and reduce_fn as kwargs. Reduces
        dimensionality of Table until only an ItemTable is left.
        """
        dim_labels = self.dimension_labels
        reduced_table = self
        for dim, reduce_fn in reduce_map.items():
            split_dims = [self.dim_dict[d] for d in dim_labels if d != dim]
            if len(split_dims):
                split_map = reduced_table.split_dimensions([dim])
                reduced_table = self.clone(None, dimensions=split_dims)
                for k, table in split_map.items():
                    reduced_table[k] = reduce_fn(table.data.values())
            else:
                data = reduce_fn(reduced_table.data.values())
                reduced_table = ItemTable({self.value.name: data},
                                          dimensions=[self.value])
        return reduced_table


    def _item_check(self, dim_vals, data):
        if not np.isscalar(data):
            raise TypeError('Table only accepts scalar values.')
        super(Table, self)._item_check(dim_vals, data)


    def viewmap(self, dimensions):
        split_dims = [dim for dim in self.dimension_labels
                      if dim not in dimensions]
        if len(dimensions) < self.ndims:
            return self.split_dimensions(split_dims, map_type=ViewMap)
        else:
            vmap = ViewMap(dimensions=dimensions)
            for k, v in self.items():
                vmap[k] = ItemTable({self.value.name: v}, dimensions=[self.value])
            return vmap


    def dim_values(self, dim):
        if dim == self.value.name:
            return self.values()
        else:
            return NdMapping.dim_values(self, dim)


    def dframe(self):
        return NdMapping.dframe(self, value_label=self.value.name)
