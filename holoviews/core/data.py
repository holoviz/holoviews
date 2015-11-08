"""
The data module provides utility classes to interface with various
data backends.
"""

from distutils.version import LooseVersion
from collections import defaultdict, Iterable
from itertools import groupby

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

import param

from .dimension import OrderedDict, Dimension
from .element import Element, NdElement
from .ndmapping import NdMapping, item_check, sorted_context
from .spaces import HoloMap
from . import util


class Columns(Element):
    """
    Columns provides a general baseclass for column based
    Element types. Through the use of utility class interfaces
    data may be supplied and stored in a range of formats.

    Data is assumed to be in a columnar data format with N
    observations and at least D columns, where D is the number
    of dimensions. Data supplied in one of the native formats
    will be retained. Alternatively the columns maybe supplied
    as a tuple or the rows as a list of tuples. If the data is
    purely numeric the data will automatically be converted to
    a numpy array, otherwise it will fall back to the specified
    data_type.

    Currently either an NdElement or a pandas DataFrame are
    supported as storage formats for heterogeneous data. An
    NdElement is a HoloViews wrapper around dictionary objects,
    which maps between the key dimensions and the value dimensions.

    The Columns class also provides various methods to transform
    the data in various ways and allows indexing and selecting
    along all dimensions.
    """

    data_type = param.ObjectSelector(default='mapping', allow_None=True,
                                     objects=['pandas', 'mapping'],
                                     doc="""
        Defines the data type used for storing non-numeric data.""")

    def __init__(self, data, **kwargs):
        data, params = ColumnarData._process_data(data, self.params(), **kwargs)
        super(Columns, self).__init__(data, **params)
        self.data = self._validate_data(self.data)


    def _validate_data(self, data):
        return self.interface.validate_data(self, data)


    def __setstate__(self, state):
        """
        Restores OrderedDict based Columns objects, converting
        them to the up-to-date NdElement format.
        """
        self.__dict__ = state
        if isinstance(self.data, OrderedDict):
            self.data = NdElement(self.data, kdims=self.kdims,
                                  vdims=self.vdims, group=self.group,
                                  label=self.label)


    def closest(self, coords):
        """
        Given single or multiple samples along the first
        key dimension will return the closest actual sample
        coordinates.
        """
        if self.ndims > 1:
            NotImplementedError("Closest method currently only "
                                "implemented for 1D Elements")

        if not isinstance(coords, list): coords = [coords]
        xs = self.dimension_values(0)
        idxs = [np.argmin(np.abs(xs-coord)) for coord in coords]
        return [xs[idx] for idx in idxs] if len(coords) > 1 else xs[idxs[0]]


    def sort(self, by=[]):
        """
        Sorts the data by the values along the supplied
        dimensions.
        """
        if not by: by = self.kdims
        sorted_columns = self.interface.sort(self, by)
        return self.clone(sorted_columns)


    def range(self, dim, data_range=True):
        """
        Computes the range of values along a supplied
        dimension, taking into account the range and
        soft_range defined on the Dimension object.
        """
        dim = self.get_dimension(dim)
        if dim.range != (None, None):
            return dim.range
        elif dim in self.dimensions():
            if len(self):
                drange = self.interface.range(self, dim)
            else:
                drange = (np.NaN, np.NaN)
        if data_range:
            soft_range = [r for r in dim.soft_range if r is not None]
            if soft_range:
                return util.max_range([drange, soft_range])
            else:
                return drange
        else:
            return dim.soft_range


    def add_dimension(self, dimension, dim_pos, dim_val, **kwargs):
        """
        Create a new object with an additional key dimensions.
        Requires the dimension name or object, the desired position
        in the key dimensions and a key value scalar or sequence of
        the same length as the existing keys.
        """
        if isinstance(dimension, str):
            dimension = Dimension(dimension)

        if dimension.name in self.kdims:
            raise Exception('{dim} dimension already defined'.format(dim=dimension.name))

        dimensions = self.kdims[:]
        dimensions.insert(dim_pos, dimension)

        data = self.interface.add_dimension(self, dimension, dim_pos, dim_val)
        return self.clone(data, kdims=dimensions)


    def select(self, selection_specs=None, **selection):
        """
        Allows selecting data by the slices, sets and scalar
        values along a particular dimension. The indices
        should be supplied as keywords mapping between
        the selected dimension and value. Additionally
        selection_specs (taking the form of a list of
        type.group.label strings, types or functions) may
        be supplied, which will ensure the selection is
        only applied if the specs match the selected object.
        """
        if selection_specs and not self.matches(selection_specs):
            return self

        data = self.interface.select(self, **selection)
        if np.isscalar(data):
            return data
        else:
            return self.clone(data)


    @property
    def interface(self):
        """
        Property that return the interface class to apply
        operations on the data.
        """
        if util.is_dataframe(self.data):
            return ColumnarDataFrame
        elif isinstance(self.data, np.ndarray):
            return ColumnarArray
        elif isinstance(self.data, NdElement):
            return ColumnarNdElement


    def reindex(self, kdims=None, vdims=None):
        """
        Create a new object with a re-ordered set of dimensions.
        Allows converting key dimensions to value dimensions
        and vice versa.
        """
        if kdims is None:
            key_dims = [d for d in self.kdims
                        if d not in vdims]
        else:
            key_dims = [self.get_dimension(k) for k in kdims]

        if vdims is None:
            val_dims = [d for d in self.vdims
                        if d not in kdims]
        else:
            val_dims = [self.get_dimension(v) for v in vdims]


        data = self.interface.reindex(self, key_dims, val_dims)
        return self.clone(data, kdims=key_dims, vdims=val_dims)


    def __getitem__(self, slices):
        """
        Allows slicing and selecting values in the Columns object.
        Supports multiple indexing modes:

           (1) Slicing and indexing along the values of each
               dimension in the columns object using either
               scalars, slices or sets of values.
           (2) Supplying the name of a dimension as the first
               argument will return the values along that
               dimension as a numpy array.
           (3) Slicing of all key dimensions and selecting
               a single value dimension by name.
           (4) A boolean array index matching the length of
               the Columns object.
        """
        if slices is (): return self
        if isinstance(slices, np.ndarray) and slices.dtype.kind == 'b':
            if not len(slices) == len(self):
                raise IndexError("Boolean index must match length of sliced object")
            return self.clone(self.data[slices])
        if not isinstance(slices, tuple): slices = (slices,)
        value_select = None
        if len(slices) == 1 and slices[0] in self.dimensions():
            return self.dimension_values(slices[0])
        elif len(slices) == self.ndims+1 and slices[self.ndims] in self.dimensions():
            selection = dict(zip(self.dimensions('key', label=True), slices))
            value_select = slices[self.ndims]
        else:
            selection = dict(zip(self.dimensions(label=True), slices))
        data = self.select(**selection)
        if value_select:
            values = data.dimension_values(value_select)
            if len(values) > 1:
                return values
            else:
                return values[0]
        return data


    def sample(self, samples=[]):
        """
        Allows sampling of Columns as an iterator of coordinates
        matching the key dimensions, returning a new object
        containing just the selected samples.
        """
        return self.clone(self.interface.sample(self, samples))


    def reduce(self, dimensions=[], function=None, **reduce_map):
        """
        Allows reducing the values along one or more key dimension
        with the supplied function. The dimensions may be supplied
        as a list and a function to apply or a mapping between the
        dimensions and functions to apply along each dimension.
        """
        reduce_dims, reduce_map = self._reduce_map(dimensions, function, reduce_map)
        reduced = self
        for reduce_fn, group in reduce_map:
            reduced = self.interface.reduce(reduced, group, function)

        if np.isscalar(reduced):
            return reduced
        else:
            kdims = [kdim for kdim in self.kdims if kdim not in reduce_dims]
            return self.clone(reduced, kdims=kdims)


    def aggregate(self, dimensions=[], function=None):
        """
        Aggregates over the supplied key dimensions with the
        defined function.
        """
        if not isinstance(dimensions, list): dimensions = [dimensions]
        if not dimensions: dimensions = self.kdims
        aggregated = self.interface.aggregate(self, dimensions, function)
        kdims = [self.get_dimension(d) for d in dimensions]
        return self.clone(aggregated, kdims=kdims)


    def groupby(self, dimensions=[], container_type=HoloMap, group_type=None, **kwargs):
        if not isinstance(dimensions, list): dimensions = [dimensions]
        if not len(dimensions): dimensions = self.dimensions('key', True)
        dimensions = [self.get_dimension(d).name for d in dimensions]
        invalid_dims = list(set(dimensions) - set(self.dimensions('key', True)))
        if invalid_dims:
            raise Exception('Following dimensions could not be found:\n%s.'
                            % invalid_dims)
        if group_type is None:
            group_type = type(self)
        return self.interface.groupby(self, dimensions, container_type, group_type, **kwargs)


    @classmethod
    def collapse_data(cls, data, function=None, kdims=None, **kwargs):
        """
        Class method utility function to concatenate the supplied data
        and apply a groupby operation along the supplied key dimensions
        then aggregates across the groups with the supplied function.
        """
        if isinstance(data[0], NdElement):
            return data[0].collapse_data(data, function, kdims, **kwargs)
        elif isinstance(data[0], np.ndarray):
            return ColumnarArray.collapse_data(data, function, kdims, **kwargs)
        elif util.is_dataframe(data[0]):
            return ColumnarDataFrame.collapse_data(data, function, kdims, **kwargs)


    @classmethod
    def concat(cls, columns_objs):
        """
        Concatenates a list of Columns objects. If data types
        don't match all types will be converted to that of
        the first object before concatenation.
        """
        columns = columns_objs[0]
        if len({col.interface for col in columns_objs}) > 1:
            if isinstance(columns.data, NdElement):
                columns_objs = [co.mapping(as_table=True) for co in columns_objs]
            elif isinstance(columns.data, np.ndarray):
                columns_objs = [co.array(as_table=True) for co in columns_objs]
            elif util.is_dataframe(data[0]):
                columns_objs = [co.dframe(as_table=True) for co in columns_objs]
        return columns.clone(columns.interface.concat(columns_objs))


    def __len__(self):
        """
        Returns the number of rows in the Columns object.
        """
        return self.interface.length(self)


    @property
    def shape(self):
        "Returns the shape of the data."
        return self.interface.shape(self)


    def dimension_values(self, dim, unique=False):
        """
        Returns the values along a particular
        dimension. If unique values are requested
        will return only unique values.
        """
        dim = self.get_dimension(dim).name
        dim_vals = self.interface.values(self, dim)
        if unique:
            return np.unique(dim_vals)
        else:
            return dim_vals


    def dframe(self, as_table=False):
        """
        Returns the data in the form of a DataFrame,
        if as_table is requested the data will be
        wrapped in a Table object.
        """
        return self.interface.dframe(self, as_table)




class ColumnarData(param.Parameterized):

    @classmethod
    def range(cls, columns, dimension):
        column = columns.dimension_values(dimension)
        if columns.get_dimension_type(dimension) is np.datetime64:
            return column.min(), column.max()
        else:
            try:
                return (np.nanmin(column), np.nanmax(column))
            except TypeError:
                column.sort()
                return column[0], column[-1]


    @classmethod
    def dframe(cls, columns, as_table=False):
        return Element.dframe(columns, as_table)


    @classmethod
    def shape(cls, columns):
        return columns.data.shape


    @classmethod
    def _process_data(cls, data, paramobjs, **kwargs):
        params = {}
        if isinstance(data, Element):
            params = util.get_param_values(data)

        if isinstance(data, NdElement):
            params['kdims'] = [d for d in params['kdims'] if d != 'Index']
        elif isinstance(data, Element):
            dimensions = data.dimensions(label=True)
            data = tuple(data.dimension_values(d) for d in data.dimensions())

        if isinstance(data, Columns):
            data = data.data
        elif util.is_dataframe(data):
            kdims, vdims = cls._process_df_dims(data, paramobjs, **kwargs)
            params['kdims'] = kdims
            params['vdims'] = vdims
        elif not isinstance(data, (NdElement, dict)):
            if isinstance(data, np.ndarray):
                array = data
            elif isinstance(data, tuple):
                try:
                    array = np.column_stack(data)
                except:
                    array = None
            else:
                data = [] if data is None else list(data)
                try:
                    array = np.array(data)
                except:
                    array = None
            # If ndim > 2 data is assumed to be a mapping
            if (isinstance(data[0], tuple) and any(isinstance(d, tuple) for d in data[0])
                or (array is not None and array.ndim > 2)):
                pass
            elif array is None or array.dtype.kind in ['S', 'U', 'O']:
                # Check if data is of non-numeric type
                # Then use defined data type
                data_type = kwargs.get('data_type', paramobjs['data_type'].default)
                kdims = kwargs.get('kdims', paramobjs['kdims'].default)
                vdims = kwargs.get('vdims', paramobjs['vdims'].default)
                if data_type == 'pandas':
                    columns = [d.name if isinstance(d, Dimension) else d
                               for d in kdims+vdims]
                    if isinstance(data, tuple):
                        data = pd.DataFrame.from_items([(c, d) for c, d in
                                                        zip(columns, data)])
                    else:
                        data = pd.DataFrame(data, columns=columns)
                else:
                    if isinstance(data, tuple):
                        data = zip(*data)
                    ndims = len(kdims)
                    data = [(tuple(row[:ndims]), tuple(row[ndims:]))
                            for row in data]
            else:
                data = array
        params.update(kwargs)
        if 'kdims' not in params:
            params['kdims'] = paramobjs['kdims'].default
        if 'vdims' not in params:
            params['vdims'] = paramobjs['vdims'].default
        if isinstance(data, (dict, list)):
            data = NdElement(data, kdims=params['kdims'],
                             vdims=params['vdims'])
        return data, params


    @classmethod
    def _process_df_dims(cls, data, paramobjs, **kwargs):
        columns = data.columns
        kdims = kwargs.get('kdims', [])
        vdims = kwargs.get('vdims', [])
        ndim = paramobjs['kdims'].bounds[1] if paramobjs['kdims'].bounds else None
        if 'kdims' in kwargs and 'vdims' not in kwargs:
            vdims = [c for c in data.columns if c not in kdims]
        elif 'kdims' not in kwargs and 'vdims' in kwargs:
            kdims = [c for c in data.columns if c not in kdims][:ndim]
        elif 'kdims' not in kwargs and 'vdims' not in kwargs:
            kdims = list(data.columns[:ndim])
            vdims = list(data.columns[ndim:])
        col_labels = [c.name if isinstance(c, Dimension) else c
                      for c in kdims+vdims]
        if not all(c in data.columns for c in col_labels):
                raise ValueError("Supplied dimensions don't match columns"
                                 "in the dataframe.")
        return kdims, vdims


    @classmethod
    def length(cls, columns):
        return len(columns.data)


    @classmethod
    def validate_data(cls, columns, data):
        return data



class ColumnarNdElement(ColumnarData):

    @classmethod
    def validate_data(cls, columns, data):
        return data

    @classmethod
    def shape(cls, columns):
        return (len(columns), len(columns.dimensions()))

    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values):
        return columns.data.add_dimension(dimension, dim_pos+1, values)

    @classmethod
    def concat(cls, columns_objs):
        return [(k[1:], v) for col in columns_objs
                for k, v in col.data.data.items()]

    @classmethod
    def sort(cls, columns, by=[]):
        if not len(by): by = columns.dimensions('key', True)
        return columns.data.sort(by)

    @classmethod
    def values(cls, columns, dim):
        return columns.data.dimension_values(dim)

    @classmethod
    def reindex(cls, columns, kdims=None, vdims=None):
        return columns.data.reindex(kdims, vdims)

    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        if 'kdims' not in kwargs:
            kwargs['kdims'] = [d for d in columns.kdims if d not in dimensions]
        with item_check(False), sorted_context(False):
            return columns.data.groupby(dimensions, container_type, group_type, **kwargs)

    @classmethod
    def select(cls, columns, **selection):
        return columns.data.select(**selection)

    @classmethod
    def collapse_data(cls, data, function, kdims=None, **kwargs):
        return data[0].collapse_data(data, function, kdims, **kwargs)

    @classmethod
    def sample(cls, columns, samples=[]):
        return columns.data.sample(samples)

    @classmethod
    def reduce(cls, columns, reduce_dims, function):
        return columns.data.reduce(columns.data, reduce_dims, function)

    @classmethod
    def aggregate(cls, columns, dimensions, function):
        return columns.data.aggregate(dimensions, function)



class ColumnarDataFrame(ColumnarData):


    @classmethod
    def range(cls, columns, dimension):
        column = columns.data[columns.get_dimension(dimension).name]
        return (column.min(), column.max())


    @classmethod
    def concat(cls, columns_objs):
        return pd.concat([col.data for col in columns_objs])


    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        index_dims = [columns.get_dimension(d) for d in dimensions]
        element_dims = [kdim for kdim in columns.kdims
                        if kdim not in index_dims]

        element_kwargs = dict(util.get_param_values(columns),
                              kdims=element_dims)
        element_kwargs.update(kwargs)
        names = [d.name for d in columns.dimensions()
                 if d not in dimensions]
        map_data = [(k, group_type(v, **element_kwargs))
                    for k, v in columns.data.groupby(dimensions)]
        with item_check(False), sorted_context(False):
            return container_type(map_data, kdims=index_dims)


    @classmethod
    def reduce(cls, columns, reduce_dims, function=None):
        """
        The aggregate function accepts either a list of Dimensions
        and a function to apply to find the aggregate across
        those Dimensions or a list of dimension/function pairs
        to apply one by one.
        """
        kdims = [kdim.name for kdim in columns.kdims if kdim not in reduce_dims]
        vdims = columns.dimensions('value', True)
        if kdims:
            reduced = columns.data.reindex(columns=kdims+vdims).\
                      groupby(kdims).aggregate(function).reset_index()
        else:
            if isinstance(function, np.ufunc):
                reduced = function.reduce(columns.data, axis=0)
            else:
                reduced = function(columns.data, axis=0)[vdims]
            if len(reduced) == 1:
                reduced = reduced[0]
            else:
                reduced = pd.DataFrame([reduced], columns=vdims)
        return reduced


    @classmethod
    def reindex(cls, columns, kdims=None, vdims=None):
        # DataFrame based tables don't need to be reindexed
        return columns.data


    @classmethod
    def collapse_data(cls, data, function, kdims, **kwargs):
        return pd.concat(data).groupby([d.name for d in kdims]).agg(function).reset_index()


    @classmethod
    def sort(cls, columns, by=[]):
        import pandas as pd
        if not isinstance(by, list): by = [by]
        if not by: by = range(columns.ndims)
        cols = [columns.get_dimension(d).name for d in by]

        if (not isinstance(columns.data, pd.DataFrame) or
            LooseVersion(pd.__version__) < '0.17.0'):
            return columns.data.sort(columns=cols)
        return columns.data.sort_values(by=cols)


    @classmethod
    def select(cls, columns, selection_specs=None, **select):
        df = columns.data
        selected_kdims = []
        mask = True
        for dim, k in select.items():
            if isinstance(k, tuple):
                k = slice(*k)
            if isinstance(k, slice):
                if k.start is not None:
                    mask &= k.start <= df[dim]
                if k.stop is not None:
                    mask &= df[dim] < k.stop
            elif isinstance(k, (set, list)):
                iter_slcs = []
                for ik in k:
                    iter_slcs.append(df[dim] == ik)
                mask &= np.logical_or.reduce(iter_slcs)
            else:
                if dim in columns.kdims: selected_kdims.append(dim)
                mask &= df[dim] == k
        df = df.ix[mask]
        if len(set(selected_kdims)) == columns.ndims:
            if len(df) and len(columns.vdims) == 1:
                df = df[columns.vdims[0].name].iloc[0]
        return df


    @classmethod
    def values(cls, columns, dim):
        data = columns.data[dim]
        if util.dd and isinstance(data, util.dd.Series):
            data = data.compute()
        return np.array(data)


    @classmethod
    def aggregate(cls, columns, dimensions, function):
        data = columns.data
        cols = [d.name for d in columns.kdims if d in dimensions]
        vdims = columns.dimensions('value', True)
        return data.reindex(columns=cols+vdims).groupby(cols).\
            aggregate(function).reset_index()


    @classmethod
    def sample(cls, columns, samples=[]):
        data = columns.data
        mask = np.zeros(cls.length(columns), dtype=bool)
        for sample in samples:
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                mask = np.logical_or(mask, data.iloc[:, i]==v)
        return data[mask]


    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values):
        data = columns.data.copy()
        data.insert(dim_pos, dimension.name, values)
        return data


    @classmethod
    def dframe(cls, columns, as_table=False):
        if as_table:
            from ..element import Table
            return Table(columns)
        return columns.data



class ColumnarArray(ColumnarData):

    @classmethod
    def validate_data(cls, columns, data):
        if data.ndim == 1:
            data = np.column_stack([np.arange(len(data)), data])
        return data


    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values):
        data = columns.data.copy()
        return np.insert(data, dim_pos, values, axis=1)


    @classmethod
    def concat(cls, columns_objs):
        return np.concatenate([col.data for col in columns_objs])


    @classmethod
    def dframe(cls, columns, as_table=False):
        return Element.dframe(columns, as_table)


    @classmethod
    def sort(cls, columns, by=[]):
        data = columns.data
        idxs = [columns.get_dimension_index(dim) for dim in by]
        return data[np.lexsort(np.flipud(data[:, idxs].T))]


    @classmethod
    def values(cls, columns, dim):
        data = columns.data
        dim_idx = columns.get_dimension_index(dim)
        if data.ndim == 1:
            data = np.atleast_2d(data).T
        return data[:, dim_idx]


    @classmethod
    def reindex(cls, columns, kdims=None, vdims=None):
        # DataFrame based tables don't need to be reindexed
        dims = kdims + vdims
        data = [columns.dimension_values(d) for d in dims]
        return np.column_stack(data)


    @classmethod
    def groupby(cls, columns, dimensions, container_type=HoloMap, group_type=None, raw=False, **kwargs):
        data = columns.data

        # Get dimension objects, labels, indexes and data
        dimensions = [columns.get_dimension(d) for d in dimensions]
        dim_idxs = [columns.get_dimension_index(d) for d in dimensions]
        ndims = len(dimensions)
        kwargs['kdims'] = [kdim for kdim in columns.kdims
                           if kdim not in dimensions]

        # Find unique entries along supplied dimensions
        # by creating a view that treats the selected
        # groupby keys as a single object.
        indices = data[:, dim_idxs].copy()
        view = indices.view(np.dtype((np.void, indices.dtype.itemsize * indices.shape[1])))
        _, idx = np.unique(view, return_index=True)
        idx.sort()
        unique_indices = indices[idx]

        params = util.get_param_values(columns)
        params.update(kwargs)

        # Iterate over the unique entries building masks
        # to apply the group selection
        grouped_data = []
        for group in unique_indices:
            mask = np.logical_and.reduce([data[:, i] == group[i]
                                         for i in range(ndims)])
            group_data = data[mask, ndims:]
            if not raw:
                if group_type is None:
                    group_data = columns.clone(group_data, **params)
                else:
                    group_data = group_type(group_data, **params)
            grouped_data.append((tuple(group), group_data))

        if raw:
            return grouped_data
        else:
            with item_check(False), sorted_context(False):
                return container_type(grouped_data, kdims=dimensions)


    @classmethod
    def select(cls, columns, **selection):
        data = columns.data
        mask = True
        selected_kdims = []
        value = selection.pop('value', None)
        for d, slc in selection.items():
            idx = columns.get_dimension_index(d)
            if isinstance(slc, tuple):
                slc = slice(*slc)
            if isinstance(slc, slice):
                if slc.start is not None:
                    mask &= slc.start <= data[:, idx]
                if slc.stop is not None:
                    mask &= data[:, idx] < slc.stop
            elif isinstance(slc, (set, list)):
                mask &= np.in1d(data[:, idx], list(slc))
            else:
                if d in columns.kdims: selected_kdims.append(d)
                if columns.ndims == 1:
                    data_index = np.argmin(np.abs(data[:, idx] - slc))
                    data = data[data_index, :]
                    break
                else:
                    mask &= data[:, idx] == slc
        if mask is not True:
            data = data[mask, :]
        data = np.atleast_2d(data)
        if len(data) and len(set(selected_kdims)) == columns.ndims:
            if len(data) == 1 and len(columns.vdims) == 1:
                data = data[0, columns.ndims]
        return data


    @classmethod
    def collapse_data(cls, data, function, kdims=None, **kwargs):
        ndims = data[0].shape[1]
        nkdims = len(kdims)
        data = data[0] if len(data) == 0 else np.concatenate(data)
        vdims = ['Value Dimension %s' % i for i in range(ndims-len(kdims))]
        joined_data = Columns(data, kdims=kdims, vdims=vdims)

        rows = []
        for k, group in cls.groupby(joined_data, kdims, raw=True):
            row = np.zeros(ndims)
            row[:nkdims] = np.array(k)
            if isinstance(function, np.ufunc):
                collapsed = function.reduce(group)
            else:
                collapsed = function(group, axis=0, **kwargs)
            row[nkdims:] = collapsed
            rows.append(row)
        return np.array(rows)


    @classmethod
    def sample(cls, columns, samples=[]):
        data = columns.data
        mask = False
        for sample in samples:
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                mask |= data[:, i]==v
        return data[mask]


    @classmethod
    def reduce(cls, columns, reduce_dims, function):
        kdims = [kdim for kdim in columns.kdims if kdim not in reduce_dims]
        if len(kdims):
            reindexed = columns.reindex(kdims)
            reduced = reindexed.collapse_data([reindexed.data], function, kdims)
        else:
            if isinstance(function, np.ufunc):
                reduced = function.reduce(columns.data, axis=0)
            else:
                reduced = function(columns.data, axis=0)
            reduced = reduced[columns.ndims:]
        if reduced.ndim == 1:
            if len(reduced) == 1:
                return reduced[0]
            else:
                return np.atleast_2d(reduced)
        return reduced


    @classmethod
    def aggregate(cls, columns, dimensions, function):
        if not isinstance(dimensions, Iterable): dimensions = [dimensions]
        rows = []
        reindexed = columns.reindex(dimensions)
        for k, group in cls.groupby(reindexed, dimensions, raw=True):
            if isinstance(function, np.ufunc):
                reduced = function.reduce(group, axis=0)
            else:
                reduced = function(group, axis=0)
            rows.append(np.concatenate([k, (reduced,) if np.isscalar(reduced) else reduced]))
        return np.array(rows)
