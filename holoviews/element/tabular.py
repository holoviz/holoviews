import numpy as np

import param

from ..core import OrderedDict, Element, Dataset, Tabular
from ..core.dimension import Dimension, dimension_name
from .selection import SelectionIndexExpr


class ItemTable(Element):
    """
    A tabular element type to allow convenient visualization of either
    a standard Python dictionary, an OrderedDict or a list of tuples
    (i.e. input suitable for an OrderedDict constructor). If an
    OrderedDict is used, the headings will be kept in the correct
    order. Tables store heterogeneous data with different labels.

    Dimension objects are also accepted as keys, allowing dimensional
    information (e.g. type and units) to be associated per heading.
    """

    kdims = param.List(default=[], bounds=(0, 0), doc="""
       ItemTables hold an index Dimension for each value they contain, i.e.
       they are equivalent to the keys.""")

    vdims = param.List(default=[Dimension('Default')], bounds=(00, None), doc="""
       ItemTables should have only index Dimensions.""")

    group = param.String(default="ItemTable", constant=True)

    @property
    def rows(self):
        return len(self.vdims)

    @property
    def cols(self):
        return 2

    def __init__(self, data, **params):
        if data is None:
            data = []
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
        str_keys = OrderedDict((dimension_name(k), v) for (k,v) in data.items())
        super(ItemTable, self).__init__(str_keys, **params)


    def __getitem__(self, heading):
        """
        Get the value associated with the given heading (key).
        """
        if heading == ():
            return self
        if heading not in self.vdims:
            raise KeyError("%r not in available headings." % heading)
        return np.array(self.data.get(heading, np.NaN))


    @classmethod
    def collapse_data(cls, data, function, **kwargs):
        param.main.param.warning(
            'ItemTable.collapse_data is deprecated and '
            'should no longer be used.')
        groups = np.vstack([np.array(odict.values()) for odict in data]).T
        return OrderedDict(zip(data[0].keys(), function(groups, axis=-1, **kwargs)))


    def dimension_values(self, dimension, expanded=True, flat=True):
        dimension = self.get_dimension(dimension, strict=True).name
        if dimension in self.dimensions('value', label=True):
            return np.array([self.data.get(dimension, np.NaN)])
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
            return self.dimensions('value')[row].pprint_label
        else:
            dim = self.get_dimension(row)
            heading = self.vdims[row]
            return dim.pprint_value(self.data.get(heading.name, np.NaN))


    def hist(self, *args, **kwargs):
        raise NotImplementedError("ItemTables are not homogeneous and "
                                  "don't support histograms.")


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        if col == 0:  return 'heading'
        else:         return 'data'



class Table(SelectionIndexExpr, Dataset, Tabular):
    """
    Table is a Dataset type, which gets displayed in a tabular
    format and is convertible to most other Element types.
    """

    group = param.String(default='Table', constant=True, doc="""
         The group is used to describe the Table.""")
