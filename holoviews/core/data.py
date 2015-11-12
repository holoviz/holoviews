"""
The data module provides utility classes to interface with various data
backends.
"""

import sys
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
    Columns provides a general baseclass for column based Element types
    that supports a range of data formats.

    Currently numpy arrays are supported for data with a uniform
    type. For storage of columns with heterogenous types, either a
    dictionary format or a pandas DataFrame may be used for storage.

    The Columns class supports various methods offering a consistent way
    of working with the stored data regardless of the storage format
    used. These operations include indexing, selection and various ways
    of aggregating or collapsing the data with a supplied function.
    """

    datatype = param.List(['array', 'dictionary', 'dataframe' ],
        doc=""" A priority list of the data types to be used for storage
        on the .data attribute. If the input supplied to the element
        constructor cannot be put into the requested format, the next
        format listed will be used until a suitable format is found (or
        the data fails to be understood).""")

    def __init__(self, data, **kwargs):
        if isinstance(data, Element):
            pvals = util.get_param_values(data)
            kwargs.update([(l, pvals[l]) for l in ['group', 'label']
                           if l in pvals and l not in kwargs])
        initialized = DataColumns.initialize(type(self), data,
                                             kwargs.get('kdims'),
                                             kwargs.get('vdims'),
                                             datatype=kwargs.get('datatype'))
        (data, kdims, vdims, self.interface) = initialized
        super(Columns, self).__init__(data, **dict(kwargs, kdims=kdims, vdims=vdims))
        self.interface.validate(self)


    def __setstate__(self, state):
        """
        Restores OrderedDict based Columns objects, converting them to
        the up-to-date NdElement format.
        """
        self.__dict__ = state
        if isinstance(self.data, OrderedDict):
            self.data = NdElement(self.data, kdims=self.kdims,
                                  vdims=self.vdims, group=self.group,
                                  label=self.label)
            self.interface = NdColumns
        elif isinstance(self.data, np.ndarray):
            self.interface = ArrayColumns
        elif util.is_dataframe(self.data):
            self.interface = DFColumns


    def closest(self, coords):
        """
        Given single or multiple samples along the first key dimension
        will return the closest actual sample coordinates.
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
        Sorts the data by the values along the supplied dimensions.
        """
        if not by: by = self.kdims
        sorted_columns = self.interface.sort(self, by)
        return self.clone(sorted_columns)


    def range(self, dim, data_range=True):
        """
        Computes the range of values along a supplied dimension, taking
        into account the range and soft_range defined on the Dimension
        object.
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
        Create a new object with an additional key dimensions.  Requires
        the dimension name or object, the desired position in the key
        dimensions and a key value scalar or sequence of the same length
        as the existing keys.
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
        Allows selecting data by the slices, sets and scalar values
        along a particular dimension. The indices should be supplied as
        keywords mapping between the selected dimension and
        value. Additionally selection_specs (taking the form of a list
        of type.group.label strings, types or functions) may be
        supplied, which will ensure the selection is only applied if the
        specs match the selected object.
        """
        if selection_specs and not self.matches(selection_specs):
            return self

        data = self.interface.select(self, **selection)
        if np.isscalar(data):
            return data
        else:
            return self.clone(data)


    def reindex(self, kdims=None, vdims=None):
        """
        Create a new object with a re-ordered set of dimensions.  Allows
        converting key dimensions to value dimensions and vice versa.
        """
        if kdims is None:
            key_dims = [d for d in self.kdims if d not in vdims]
        else:
            key_dims = [self.get_dimension(k) for k in kdims]

        if vdims is None:
            val_dims = [d for d in self.vdims if d not in kdims]
        else:
            val_dims = [self.get_dimension(v) for v in vdims]

        data = self.interface.reindex(self, key_dims, val_dims)
        return self.clone(data, kdims=key_dims, vdims=val_dims)


    def __getitem__(self, slices):
        """
        Allows slicing and selecting values in the Columns object.
        Supports multiple indexing modes:

           (1) Slicing and indexing along the values of each dimension
               in the columns object using either scalars, slices or
               sets of values.
           (2) Supplying the name of a dimension as the first argument
               will return the values along that dimension as a numpy
               array.
           (3) Slicing of all key dimensions and selecting a single
               value dimension by name.
           (4) A boolean array index matching the length of the Columns
               object.
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
        matching the key dimensions, returning a new object containing
        just the selected samples.
        """
        return self.clone(self.interface.sample(self, samples))


    def reduce(self, dimensions=[], function=None, **reduce_map):
        """
        Allows reducing the values along one or more key dimension with
        the supplied function. The dimensions may be supplied as a list
        and a function to apply or a mapping between the dimensions and
        functions to apply along each dimension.
        """
        if any(dim in self.vdims for dim in dimensions):
            raise Exception("Reduce cannot be applied to value dimensions")
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
        Aggregates over the supplied key dimensions with the defined
        function.
        """
        if function is None:
            raise ValueError("The aggregate method requires a function to be specified")
        if not isinstance(dimensions, list): dimensions = [dimensions]
        if not dimensions: dimensions = self.kdims
        aggregated = self.interface.aggregate(self, dimensions, function)
        kdims = [self.get_dimension(d) for d in dimensions]
        return self.clone(aggregated, kdims=kdims)


    def groupby(self, dimensions=[], container_type=HoloMap, group_type=None, **kwargs):
        """
        Return the results of a groupby operation over the specified
        dimensions as an object of type container_type (expected to be
        dictionary-like).

        Keys vary over the columns (dimensions) and the corresponding
        values are collections of group_type (e.g list, tuple)
        constructed with kwargs (if supplied).
        """
        if not isinstance(dimensions, list): dimensions = [dimensions]
        if not len(dimensions): dimensions = self.dimensions('key', True)
        if group_type is None: group_type = type(self)

        dimensions = [self.get_dimension(d).name for d in dimensions]
        invalid_dims = list(set(dimensions) - set(self.dimensions('key', True)))
        if invalid_dims:
            raise Exception('Following dimensions could not be found:\n%s.'
                            % invalid_dims)
        return self.interface.groupby(self, dimensions, container_type,
                                      group_type, **kwargs)

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
        Returns the values along a particular dimension. If unique
        values are requested will return only unique values.
        """
        dim = self.get_dimension(dim).name
        dim_vals = self.interface.values(self, dim)
        if unique:
            return np.unique(dim_vals)
        else:
            return dim_vals


    def dframe(self, dimensions=None):
        """
        Returns the data in the form of a DataFrame.
        """
        if dimensions:
            dimensions = [self.get_dimension(d).name for d in dimensions]
        return self.interface.dframe(self, dimensions)

    def columns(self, dimensions=None):
        if dimensions is None: dimensions = self.dimensions()
        dimensions = [self.get_dimension(d) for d in dimensions]
        return {d.name: self.dimension_values(d) for d in dimensions}



class DataColumns(param.Parameterized):

    interfaces = {}

    datatype = None

    @classmethod
    def register(cls, interface):
        cls.interfaces[interface.datatype] = interface


    @classmethod
    def cast(cls, columns, datatype=None, cast_type=None):
        """
        Given a list of Columns objects, cast them to the specified
        datatype (by default the format matching the current interface)
        with the given cast_type (if specified).
        """
        classes = {type(c) for c in columns}
        if len(classes) > 1:
            raise Exception("Please supply the common cast type")
        else:
            cast_type = classes.pop()

        if datatype is None:
           datatype = cls.datatype

        unchanged = all({c.interface==cls for c in columns})
        if unchanged and set([cast_type])==classes:
            return columns
        elif unchanged:
            return [cast_type(co, **dict(util.get_param_values(co)) ) for co in columns]

        return [cast_type(co.columns(), datatype=[datatype],
                          **dict(util.get_param_values(co))) for co in columns]


    @classmethod
    def initialize(cls, eltype, data, kdims, vdims, datatype=None):
        # Process params and dimensions
        if isinstance(data, Element):
            pvals = util.get_param_values(data)
            kdims = pvals.get('kdims') if kdims is None else kdims
            vdims = pvals.get('vdims') if vdims is None else vdims

        # Process Element data
        if isinstance(data, NdElement):
            pass
        elif isinstance(data, Columns):
            data = data.data
        elif isinstance(data, Element):
            data = tuple(data.dimension_values(d) for d in kdims+vdims)
        elif (not (util.is_dataframe(data) or isinstance(data, (tuple, dict, list)))
              and sys.version_info.major >= 3):
            data = list(data)

        # Set interface priority order
        if datatype is None:
            datatype = eltype.datatype
        prioritized = [cls.interfaces[p] for p in datatype]

        head = [intfc for intfc in prioritized if type(data) in intfc.types]
        if head:
            # Prioritize interfaces which have matching types
            prioritized = head + [el for el in prioritized if el != head[0]]

        # Iterate over interfaces until one can interpret the input
        for interface in prioritized:
            try:
                (data, kdims, vdims) = interface.reshape(eltype, data, kdims, vdims)
                break
            except:
                pass
        else:
            raise ValueError("None of the available storage backends "
                             "were able to support the supplied data format.")

        return data, kdims, vdims, interface


    @classmethod
    def select_mask(cls, columns, selection):
        """
        Given a Columns object and a dictionary with dimension keys and
        selection keys (i.e tuple ranges, slices, sets, lists or literals)
        return a boolean mask over the rows in the Columns object that
        have been selected.
        """
        mask = np.ones(len(columns), dtype=np.bool)
        for dim, k in selection.items():
            if isinstance(k, tuple):
                k = slice(*k)
            arr = cls.values(columns, dim)
            if isinstance(k, slice):
                if k.start is not None:
                    mask &= k.start <= arr
                if k.stop is not None:
                    mask &= arr < k.stop
            elif isinstance(k, (set, list)):
                iter_slcs = []
                for ik in k:
                    iter_slcs.append(arr == ik)
                mask &= np.logical_or.reduce(iter_slcs)
            else:
                index_mask = arr == k
                if columns.ndims == 1 and np.sum(index_mask) == 0:
                    data_index = np.argmin(np.abs(arr - k))
                    mask = np.zeros(len(columns), dtype=np.bool)
                    mask[data_index] = True
                else:
                    mask &= index_mask
        return mask


    @classmethod
    def indexed(cls, columns, selection):
        """
        Given a Columns object and selection to be applied returns
        boolean to indicate whether a scalar value has been indexed.
        """
        selected = list(selection.keys())
        all_scalar = all(not isinstance(sel, (tuple, slice, set, list))
                         for sel in selection.values())
        all_kdims = all(d in selected for d in columns.kdims)
        return all_scalar and all_kdims and len(columns.vdims) == 1


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
    def concatenate(cls, columns, datatype=None):
        """
        Utility function to concatenate a list of Column objects,
        returning a new Columns object. Note that this is unlike the
        .concat method which only concatenates the data.
        """
        if len(set(type(c) for c in columns)) != 1:
               raise Exception("All inputs must be same type in order to concatenate")

        interfaces = set(c.interface for c in columns)
        if len(interfaces)!=1 and datatype is None:
            raise Exception("Please specify the concatenated datatype")
        elif len(interfaces)!=1:
            interface = cls.interfaces[datatype]
        else:
            interface = interfaces.pop()

        concat_data = interface.concat(columns)
        return columns[0].clone(concat_data)


    @classmethod
    def array(cls, columns, dimensions):
        return Element.array(columns, dimensions)

    @classmethod
    def dframe(cls, columns, dimensions):
        return Element.dframe(columns, dimensions)

    @classmethod
    def columns(cls, columns, dimensions):
        return Element.columns(columns, dimensions)

    @classmethod
    def shape(cls, columns):
        return columns.data.shape

    @classmethod
    def length(cls, columns):
        return len(columns.data)

    @classmethod
    def validate(cls, columns):
        pass



class NdColumns(DataColumns):

    types = (NdElement,)

    datatype = 'dictionary'

    @classmethod
    def reshape(cls, eltype, data, kdims, vdims):
        if isinstance(data, NdElement):
            kdims = [d for d in kdims if d != 'Index']
        else:
            element_params = eltype.params()
            kdims = kdims if kdims else element_params['kdims'].default
            vdims = vdims if vdims else element_params['vdims'].default

        if isinstance(data, dict) and all(d in data for d in kdims+vdims):
            data = tuple(data.get(d.name if isinstance(d, Dimension) else d)
                         for d in dimensions)

        if not isinstance(data, (NdElement, dict)):
            # If ndim > 2 data is assumed to be a mapping
            if (isinstance(data[0], tuple) and any(isinstance(d, tuple) for d in data[0])):
                pass
            else:
                if isinstance(data, tuple):
                    data = zip(*data)
                ndims = len(kdims)
                data = [(tuple(row[:ndims]), tuple(row[ndims:]))
                        for row in data]
        if isinstance(data, (dict, list)):
            data = NdElement(data, kdims=kdims, vdims=vdims)
        elif not isinstance(data, NdElement):
            raise ValueError("NdColumns interface couldn't convert data.""")
        return data, kdims, vdims


    @classmethod
    def shape(cls, columns):
        return (len(columns), len(columns.dimensions()))

    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values):
        return columns.data.add_dimension(dimension, dim_pos+1, values)

    @classmethod
    def concat(cls, columns_objs):
        cast_objs = cls.cast(columns_objs)
        return [(k[1:], v) for col in cast_objs for k, v in col.data.data.items()]

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



class DFColumns(DataColumns):

    types = (pd.DataFrame if pd else None,)

    datatype = 'dataframe'

    @classmethod
    def reshape(cls, eltype, data, kdims, vdims):
        element_params = eltype.params()
        kdim_param = element_params['kdims']
        vdim_param = element_params['vdims']
        if util.is_dataframe(data):
            columns = data.columns
            ndim = kdim_param.bounds[1] if kdim_param.bounds else None
            if kdims and not vdims:
                vdims = [c for c in data.columns if c not in kdims]
            elif vdims and not kdims:
                kdims = [c for c in data.columns if c not in kdims][:ndim]
            elif not kdims and not vdims:
                kdims = list(data.columns[:ndim])
                vdims = list(data.columns[ndim:])
        else:
            # Check if data is of non-numeric type
            # Then use defined data type
            kdims = kdims if kdims else kdim_param.default
            vdims = vdims if vdims else vdim_param.default
            columns = [d.name if isinstance(d, Dimension) else d
                       for d in kdims+vdims]

            if isinstance(data, dict):
                data = OrderedDict([(d.name if isinstance(d, Dimension) else d, v)
                                    for d, v in data.items()])
            if isinstance(data, tuple):
                data = pd.DataFrame.from_items([(c, d) for c, d in
                                                zip(columns, data)])
            else:
                data = pd.DataFrame(data, columns=columns)
        return data, kdims, vdims


    @classmethod
    def validate(cls, columns):
        if not all(c in columns.data.columns for c in columns.dimensions(label=True)):
            raise ValueError("Supplied dimensions don't match columns "
                             "in the dataframe.")


    @classmethod
    def range(cls, columns, dimension):
        column = columns.data[columns.get_dimension(dimension).name]
        return (column.min(), column.max())


    @classmethod
    def concat(cls, columns_objs):
        cast_objs = cls.cast(columns_objs)
        return pd.concat([col.data for col in cast_objs])


    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        index_dims = [columns.get_dimension(d) for d in dimensions]
        element_dims = [kdim for kdim in columns.kdims
                        if kdim not in index_dims]

        group_kwargs = {}
        if group_type != 'raw' and issubclass(group_type, Element):
            group_kwargs = dict(util.get_param_values(columns),
                                kdims=element_dims)
        group_kwargs.update(kwargs)

        data = [(k, group_type(v, **group_kwargs)) for k, v in
                columns.data.groupby(dimensions)]
        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(data, kdims=index_dims)
        else:
            return container_type(data)


    @classmethod
    def reduce(cls, columns, reduce_dims, function=None):
        """
        The aggregate function accepts either a list of Dimensions and a
        function to apply to find the aggregate across those Dimensions
        or a list of dimension/function pairs to apply one by one.
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
    def select(cls, columns, **selection):
        df = columns.data
        mask = cls.select_mask(columns, selection)
        indexed = cls.indexed(columns, selection)
        df = df.ix[mask]
        if indexed and len(df) == 1:
            return df[columns.vdims[0].name].iloc[0]
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
    def dframe(cls, columns, dimensions):
        if dimensions:
            return columns.reindex(columns=dimensions)
        else:
            return columns.data



class ArrayColumns(DataColumns):

    types = (np.ndarray,)

    datatype = 'array'

    @classmethod
    def reshape(cls, eltype, data, kdims, vdims):
        if isinstance(data, dict):
            dimensions = kdims + vdims
            if all(d in data for d in dimensions):
                columns = [data.get(d.name if isinstance(d, Dimension) else d)
                           for d in dimensions]
                data = np.column_stack(columns)
        elif isinstance(data, tuple):
            try:
                data = np.column_stack(data)
            except:
                data = None
        elif not isinstance(data, np.ndarray):
            data = np.array([], ndmin=2).T if data is None else list(data)
            try:
                data = np.array(data)
            except:
                data = None

        if data is None or data.ndim > 2 or data.dtype.kind in ['S', 'U', 'O']:
            raise ValueError("ArrayColumns interface could not handle input type.")
        elif data.ndim == 1:
            data = np.column_stack([np.arange(len(data)), data])

        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims
        return data, kdims, vdims


    @classmethod
    def array(cls, columns, dimensions):
        if dimensions:
            return Element.dframe(columns, dimensions)
        else:
            return columns.data


    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values):
        data = columns.data.copy()
        return np.insert(data, dim_pos, values, axis=1)


    @classmethod
    def concat(cls, columns_objs):
        cast_objs = cls.cast(columns_objs)
        return np.concatenate([col.data for col in cast_objs])


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
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        data = columns.data

        # Get dimension objects, labels, indexes and data
        dimensions = [columns.get_dimension(d) for d in dimensions]
        dim_idxs = [columns.get_dimension_index(d) for d in dimensions]
        ndims = len(dimensions)
        kdims = [kdim for kdim in columns.kdims
                 if kdim not in dimensions]
        vdims = columns.vdims

        # Find unique entries along supplied dimensions
        # by creating a view that treats the selected
        # groupby keys as a single object.
        indices = data[:, dim_idxs].copy()
        group_shape = indices.dtype.itemsize * indices.shape[1]
        view = indices.view(np.dtype((np.void, group_shape)))
        _, idx = np.unique(view, return_index=True)
        idx.sort()
        unique_indices = indices[idx]

        # Get group
        group_kwargs = {}
        if group_type != 'raw' and issubclass(group_type, Element):
            group_kwargs.update(util.get_param_values(columns))
            group_kwargs['kdims'] = kdims
        group_kwargs.update(kwargs)

        # Iterate over the unique entries building masks
        # to apply the group selection
        grouped_data = []
        for group in unique_indices:
            mask = np.logical_and.reduce([data[:, idx] == group[i]
                                          for i, idx in enumerate(dim_idxs)])
            group_data = data[mask, ndims:]
            if not group_type == 'raw':
                if issubclass(group_type, dict):
                    group_data = {d.name: group_data[:, i] for i, d in
                                  enumerate(kdims+vdims)}
                else:
                    group_data = group_type(group_data, **group_kwargs)
            grouped_data.append((tuple(group), group_data))

        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(grouped_data, kdims=dimensions)
        else:
            return container_type(grouped_data)


    @classmethod
    def select(cls, columns, **selection):
        mask = cls.select_mask(columns, selection)
        indexed = cls.indexed(columns, selection)
        data = np.atleast_2d(columns.data[mask, :])
        if len(data) == 1 and indexed:
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
        for k, group in cls.groupby(joined_data, kdims, list, 'raw'):
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
            reduced = cls.collapse_data([reindexed.data], function, kdims)
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
        for k, group in cls.groupby(reindexed, dimensions, list, 'raw'):
            if isinstance(function, np.ufunc):
                reduced = function.reduce(group, axis=0)
            else:
                reduced = function(group, axis=0)
            rows.append(np.concatenate([k, (reduced,) if np.isscalar(reduced) else reduced]))
        return np.array(rows)


# Register available interfaces
DataColumns.register(ArrayColumns)
DataColumns.register(NdColumns)
if pd:
    DataColumns.register(DFColumns)

