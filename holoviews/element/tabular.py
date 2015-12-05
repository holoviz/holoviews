import numpy as np

import param

from ..core import (OrderedDict, Dimension, Element, Columns,
                    Tabular, NdElement, HoloMap)


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
        if heading not in self.vdims:
            raise KeyError("%r not in available headings." % heading)
        return np.array(self.data.get(heading, np.NaN))


    @classmethod
    def collapse_data(cls, data, function, **kwargs):
        groups = np.vstack([np.array(odict.values()) for odict in data]).T
        return OrderedDict(zip(data[0].keys(), function(groups, axis=-1, **kwargs)))


    def dimension_values(self, dimension):
        dimension = self.get_dimension(dimension).name
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
            return str(self.dimensions('value')[row])
        else:
            dim = self.get_dimension(row)
            heading = self.vdims[row]
            return dim.pprint_value(self.data.get(heading.name, np.NaN))


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
        return tuple(self.data.get(d.name, np.NaN)
                     for d in self.vdims)



class Table(Columns, Tabular):
    """
    Table is an NdElement type, which gets displayed in a tabular
    format and is convertible to most other Element types.
    """

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

    def _conversion(self, kdims=None, vdims=None, mdims=None, new_type=None, sort=False, **kwargs):
        if kdims is None:
            kdims = self._table.kdims
        elif kdims and not isinstance(kdims, list): kdims = [kdims]
        if vdims is None:
            vdims = self._table.vdims
        if mdims is None:
            mdims = [d for d in self._table.kdims if d not in kdims]
        if vdims and not isinstance(vdims, list): vdims = [vdims]

        selected = self._table.reindex(mdims+kdims, vdims)
        params = {'kdims': [selected.get_dimension(kd) for kd in kdims],
                  'vdims': [selected.get_dimension(vd) for vd in vdims],
                  'label': selected.label}
        if selected.group != selected.params()['group'].default:
            params['group'] = selected.group
        params.update(kwargs)
        if len(kdims) == selected.ndims:
            element = new_type(selected, **params)
            return element.sort() if sort else element
        group = selected.groupby(mdims, container_type=HoloMap, group_type=new_type, **params)
        if sort:
            return group.map(lambda x: x.sort(), [new_type])
        else:
            return group

    def bars(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart import Bars
        return self._conversion(kdims, vdims, mdims, Bars, **kwargs)

    def bivariate(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from ..interface.seaborn import Bivariate
        return self._convert(kdims, vdims, mdims, Bivariate, **kwargs)

    def curve(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart import Curve
        return self._conversion(kdims, vdims, mdims, Curve, sort=True, **kwargs)

    def errorbars(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart import ErrorBars
        return self._conversion(kdims, vdims, mdims, ErrorBars, sort=True, **kwargs)

    def distribution(self, dim, mdims=[], **kwargs):
        from ..interface.seaborn import Distribution
        if mdims:
            reindexed = self._table.reindex(mdims+[dim])
            return reindexed.groupby(mdims, HoloMap, Distribution, **kwargs)
        else:
            table = self._table
            params = dict(kdims=[table.get_dimension(dim)],
                          label=table.label)
            if table.group != table.params()['group'].default:
                params['group'] = table.group
            return Distribution((table.dimension_values(dim),),
                                **dict(params, **kwargs))

    def heatmap(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .raster import HeatMap
        return self._conversion(kdims, vdims, mdims, HeatMap, **kwargs)

    def points(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart import Points
        return self._conversion(kdims, vdims, mdims, Points, **kwargs)

    def raster(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .raster import Raster
        heatmap = self.heatmap(kdims, vdims, **kwargs)
        return Raster(heatmap.data, **dict(self._table.get_param_values(onlychanged=True)))

    def regression(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from ..interface.seaborn import Regression
        return self._convert(kdims, vdims, mdims, Regression, **kwargs)

    def scatter(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart import Scatter
        return self._conversion(kdims, vdims, mdims, Scatter, **kwargs)

    def scatter3d(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart3d import Scatter3D
        return self._conversion(kdims, vdims, mdims, Scatter3D, **kwargs)

    def spread(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart import Spread
        return self._conversion(kdims, vdims, mdims, Spread, sort=True, **kwargs)

    def surface(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart3d import Surface
        heatmap = self.heatmap(kdims, vdims, **kwargs)
        return Surface(heatmap.data, **dict(self._table.get_param_values(onlychanged=True)))

    def trisurface(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart3d import Trisurface
        return self._conversion(kdims, vdims, mdims, Trisurface, **kwargs)

    def vectorfield(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from .chart import VectorField
        return self._conversion(kdims, vdims, VectorField, **kwargs)

