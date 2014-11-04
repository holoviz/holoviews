from collections import OrderedDict, defaultdict

import numpy as np

import param

from ..core import Dimension, Layer, NdMapping


class ItemTable(Layer):
    """
    A tabular view type to allow convenient visualization of either a
    standard Python dictionary or an OrderedDict. If an OrderedDict is
    used, the headings will be kept in the correct order. Tables store
    heterogeneous data with different labels. Optionally a list of
    dimensions corresponding to the labels can be supplied.
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


    def __init__(self, data, **kwargs):
        # Assume OrderedDict if not a vanilla Python dict
        headings = data.keys()
        if type(data) == dict:
            headings = sorted(headings)
            data = OrderedDict((h, data[h]) for h in headings)
        if 'dimensions' not in kwargs:
            kwargs['dimensions'] = headings
        super(ItemTable, self).__init__(data=data, **kwargs)


    def __getitem__(self, heading):
        """
        Get the value associated with the given heading (key).
        """
        if heading is ():
            return self
        if heading not in self.dim_dict:
            raise IndexError("%r not in available headings." % heading)
        return self.data[heading]


    def sample(self, samples=None):
        if callable(samples):
            sampled_data = OrderedDict(item for item in self.data.items()
                                       if samples(item))
        else:
            sampled_data = OrderedDict((s, self.data[s]) for s in samples)
        return self.clone(sampled_data)


    def reduce(self, **reduce_map):
        raise NotImplementedError('Tables are for heterogeneous data, which'
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
        return DataFrame({k: [v] for k, v in self.data.items()})


class Table(ItemTable, NdMapping):

    value = param.ClassSelector(class_=Dimension,
                                default=Dimension('Value'), doc="""
        The dimension description of the data held in the data array.""")

    def __init__(self, data, **params):
        super(Table, self).__init__(data, **params)
        self._data = self.data

    @property
    def rows(self):
        return len(self) + 1

    @property
    def cols(self):
        return self.ndims + 1


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
            return self.keys()[row-1][col]
            heading = list(self.dim_dict.keys())[row]
            return self.data[heading]


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        if col == self.ndims and row > 0:  return 'heading'
        else:         return 'data'


    def sample(self, samples=[]):
        """
        Allows sampling of the Table with a list of samples.
        """
        sample_data = OrderedDict()
        for sample in samples:
            sample_data[sample] = self[sample]
        return Table(sample_data, **dict(self.get_param_values()))


    def reduce(self, label_prefix='', **reduce_map):
        """
        Allows collapsing the Table down to an ItemTable View
        with a single entry.
        """
        reduced_data = OrderedDict()
        value = self.value(' '.join([label_prefix, self.value.name]))
        for dimension, reduce_fn in reduce_map.items():
            reduced_data[value] = reduce_fn(self.values())
        return ItemTable(reduced_data, title=self.title, value=self.value(value))


    def _item_check(self, dim_vals, data):
        if not np.isscalar(data):
            raise TypeError('Table only accepts scalar values.')
        super(Table, self)._item_check(dim_vals, data)


    def dframe(self):
        return NdMapping.dframe(self)