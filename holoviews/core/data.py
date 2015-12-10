"""
The data module provides utility classes to interface with various data
backends.
"""

import sys
from distutils.version import LooseVersion
from collections import defaultdict, Iterable, OrderedDict
from itertools import groupby, compress

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

from .dimension import Dimension
from .element import Element, NdElement
from .dimension import OrderedDict as cyODict
from .ndmapping import NdMapping, item_check, sorted_context
from .spaces import HoloMap
from . import util
from .util import wrap_tuple, basestring


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

    datatype = param.List(['array', 'dictionary', 'dataframe', 'ndelement'],
        doc=""" A priority list of the data types to be used for storage
        on the .data attribute. If the input supplied to the element
        constructor cannot be put into the requested format, the next
        format listed will be used until a suitable format is found (or
        the data fails to be understood).""")

    # In the 1D case the interfaces should not automatically add x-values
    # to supplied data
    _1d = False

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
            self.data = Columns(self.data, kdims=self.kdims,
                                vdims=self.vdims, group=self.group,
                                label=self.label)
            self.interface = NdColumns
        elif isinstance(self.data, np.ndarray):
            self.interface = ArrayColumns
        elif util.is_dataframe(self.data):
            self.interface = DFColumns

        super(Columns, self).__setstate__(state)

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


    def add_dimension(self, dimension, dim_pos, dim_val, vdim=False, **kwargs):
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

        if vdim:
            dims = self.vdims[:]
            dims.insert(dim_pos, dimension)
            dimensions = dict(vdims=dims)
            dim_pos += self.ndims
        else:
            dims = self.kdims[:]
            dims.insert(dim_pos, dimension)
            dimensions = dict(kdims=dims)

        data = self.interface.add_dimension(self, dimension, dim_pos, dim_val, vdim)
        return self.clone(data, **dimensions)


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
        slices = util.process_ellipses(self, slices, vdim_selection=True)
        if isinstance(slices, np.ndarray) and slices.dtype.kind == 'b':
            if not len(slices) == len(self):
                raise IndexError("Boolean index must match length of sliced object")
            return self.clone(self.interface.select(self, selection_mask=slices))
        elif slices in [(), Ellipsis]:
            return self
        if not isinstance(slices, tuple): slices = (slices,)
        value_select = None
        if len(slices) == 1 and slices[0] in self.dimensions():
            return self.dimension_values(slices[0])
        elif len(slices) == self.ndims+1 and slices[self.ndims] in self.dimensions():
            selection = dict(zip(self.dimensions('key', label=True), slices))
            value_select = slices[self.ndims]
        elif len(slices) == self.ndims+1 and isinstance(slices[self.ndims],
                                                        (Dimension,str)):
            raise Exception("%r is not an available value dimension'" % slices[self.ndims])
        else:
            selection = dict(zip(self.dimensions(label=True), slices))
        data = self.select(**selection)
        if value_select:
            if len(data) == 1:
                return data[value_select][0]
            else:
                return data.reindex(vdims=[value_select])
        return data


    def sample(self, samples=[]):
        """
        Allows sampling of Columns as an iterator of coordinates
        matching the key dimensions, returning a new object containing
        just the selected samples.
        """
        return self.clone(self.interface.sample(self, samples))


    def reduce(self, dimensions=[], function=None, spreadfn=None, **reduce_map):
        """
        Allows reducing the values along one or more key dimension with
        the supplied function. The dimensions may be supplied as a list
        and a function to apply or a mapping between the dimensions and
        functions to apply along each dimension.
        """
        if any(dim in self.vdims for dim in dimensions):
            raise Exception("Reduce cannot be applied to value dimensions")
        function, dims = self._reduce_map(dimensions, function, reduce_map)
        dims = [d for d in self.kdims if d not in dims]
        return self.aggregate(dims, function, spreadfn)


    def aggregate(self, dimensions=[], function=None, spreadfn=None, **kwargs):
        """
        Aggregates over the supplied key dimensions with the defined
        function.
        """
        if function is None:
            raise ValueError("The aggregate method requires a function to be specified")
        if not isinstance(dimensions, list): dimensions = [dimensions]
        aggregated = self.interface.aggregate(self, dimensions, function, **kwargs)
        aggregated = self.interface.unpack_scalar(self, aggregated)

        kdims = [self.get_dimension(d) for d in dimensions]
        vdims = self.vdims
        if spreadfn:
            error = self.interface.aggregate(self, dimensions, spreadfn)
            spread_name = spreadfn.__name__
            ndims = len(vdims)
            error = self.clone(error, kdims=kdims)
            combined = self.clone(aggregated, kdims=kdims)
            for i, d in enumerate(vdims):
                dim = d('_'.join([d.name, spread_name]))
                combined = combined.add_dimension(dim, ndims+i, error[d], True)
            return combined

        if np.isscalar(aggregated):
            return aggregated
        else:
            return self.clone(aggregated, kdims=kdims, vdims=vdims)



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
            kdims = [kdim for kdim in kdims if kdim != 'Index']
        elif isinstance(data, Columns):
            data = data.data
        elif isinstance(data, Element):
            data = tuple(data.dimension_values(d) for d in kdims+vdims)
        elif (not (util.is_dataframe(data) or isinstance(data, (tuple, dict, np.ndarray, list)))
              and sys.version_info.major >= 3):
            data = list(data)

        # Set interface priority order
        if datatype is None:
            datatype = eltype.datatype
        prioritized = [cls.interfaces[p] for p in datatype
                       if p in cls.interfaces]

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
    def reduce(cls, columns, reduce_dims, function, **kwargs):
        kdims = [kdim for kdim in columns.kdims if kdim not in reduce_dims]
        return cls.aggregate(columns, kdims, function, **kwargs)

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

    datatype = 'ndelement'

    @classmethod
    def reshape(cls, eltype, data, kdims, vdims):
        if isinstance(data, NdElement):
            kdims = [d for d in kdims if d != 'Index']
        else:
            element_params = eltype.params()
            kdims = kdims if kdims else element_params['kdims'].default
            vdims = vdims if vdims else element_params['vdims'].default

        dimensions = [d.name if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if ((isinstance(data, dict) or util.is_dataframe(data)) and
            all(d in data for d in dimensions)):
            data = tuple(data.get(d) for d in dimensions)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                if eltype._1d:
                    data = np.atleast_2d(data).T
                else:
                    data = (np.arange(len(data)), data)
            else:
                data = tuple(data[:, i]  for i in range(data.shape[1]))
        elif isinstance(data, list) and np.isscalar(data[0]):
            data = (np.arange(len(data)), data)

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
    def add_dimension(cls, columns, dimension, dim_pos, values, vdim):
        return columns.data.add_dimension(dimension, dim_pos+1, values, vdim)

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
    def select(cls, columns, selection_mask=None, **selection):
        if selection_mask is None:
            return columns.data.select(**selection)
        else:
            return columns.data[selection_mask]

    @classmethod
    def sample(cls, columns, samples=[]):
        return columns.data.sample(samples)

    @classmethod
    def reduce(cls, columns, reduce_dims, function, **kwargs):
        return columns.data.reduce(columns.data, reduce_dims, function)

    @classmethod
    def aggregate(cls, columns, dimensions, function, **kwargs):
        return columns.data.aggregate(dimensions, function, **kwargs)

    @classmethod
    def unpack_scalar(cls, columns, data):
        if len(data) == 1 and len(data.kdims) == 1 and len(data.vdims) == 1:
            return list(data.data.values())[0][0]
        else:
            return data



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
            ndim = len(kdim_param.default) if kdim_param.bounds else None
            if kdims and not vdims:
                vdims = [c for c in data.columns if c not in kdims]
            elif vdims and not kdims:
                kdims = [c for c in data.columns if c not in vdims][:ndim]
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

            if ((isinstance(data, dict) and all(c in data for c in columns)) or
                (isinstance(data, NdElement) and all(c in data.dimensions() for c in columns))):
                data = OrderedDict(((d, data[d]) for d in columns))
            elif isinstance(data, dict) and not all(d in data for d in columns):
                column_data = zip(*((wrap_tuple(k)+wrap_tuple(v))
                                    for k, v in data.items()))
                data = OrderedDict(((c, col) for c, col in zip(columns, column_data)))
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    if eltype._1d:
                        data = np.atleast_2d(data).T
                    else:
                        data = (range(len(data)), data)
                else:
                    data = tuple(data[:, i]  for i in range(data.shape[1]))

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
                columns.data.groupby(dimensions, sort=False)]
        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(data, kdims=index_dims)
        else:
            return container_type(data)


    @classmethod
    def aggregate(cls, columns, dimensions, function, **kwargs):
        data = columns.data
        cols = [d.name for d in columns.kdims if d in dimensions]
        vdims = columns.dimensions('value', True)
        reindexed = data.reindex(columns=cols+vdims)
        if len(dimensions):
            return reindexed.groupby(cols, sort=False).aggregate(function, **kwargs).reset_index()
        else:
            agg = reindexed.apply(function, **kwargs)
            return pd.DataFrame.from_items([(col, [v]) for col, v in
                                            zip(agg.index, agg.values)])


    @classmethod
    def unpack_scalar(cls, columns, data):
        """
        Given a columns object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if len(data) != 1 or len(data.columns) > 1:
            return data
        return data.iat[0,0]


    @classmethod
    def reindex(cls, columns, kdims=None, vdims=None):
        # DataFrame based tables don't need to be reindexed
        return columns.data


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
    def select(cls, columns, selection_mask=None, **selection):
        df = columns.data
        if selection_mask is None:
            selection_mask = cls.select_mask(columns, selection)
        indexed = cls.indexed(columns, selection)
        df = df.ix[selection_mask]
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
    def sample(cls, columns, samples=[]):
        data = columns.data
        mask = False
        for sample in samples:
            sample_mask = True
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                sample_mask = np.logical_and(sample_mask, data.iloc[:, i]==v)
            mask |= sample_mask
        return data[mask]


    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values, vdim):
        data = columns.data.copy()
        data.insert(dim_pos, dimension.name, values)
        return data


    @classmethod
    def dframe(cls, columns, dimensions):
        if dimensions:
            return columns.reindex(columns=dimensions)
        else:
            return columns.data.copy()



class ArrayColumns(DataColumns):

    types = (np.ndarray,)

    datatype = 'array'

    @classmethod
    def reshape(cls, eltype, data, kdims, vdims):
        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims

        dimensions = [d.name if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if ((isinstance(data, dict) or util.is_dataframe(data)) and
            all(d in data for d in dimensions)):
            columns = [data[d] for d in dimensions]
            data = np.column_stack(columns)
        elif isinstance(data, dict) and not all(d in data for d in dimensions):
            columns = zip(*((wrap_tuple(k)+wrap_tuple(v))
                            for k, v in data.items()))
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
            if eltype._1d:
                data = np.atleast_2d(data).T
            else:
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
    def add_dimension(cls, columns, dimension, dim_pos, values, vdim):
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
    def select(cls, columns, selection_mask=None, **selection):
        if selection_mask is None:
            selection_mask = cls.select_mask(columns, selection)
        indexed = cls.indexed(columns, selection)
        data = np.atleast_2d(columns.data[selection_mask, :])
        if len(data) == 1 and indexed:
            data = data[0, columns.ndims]
        return data



    @classmethod
    def sample(cls, columns, samples=[]):
        data = columns.data
        mask = False
        for sample in samples:
            sample_mask = True
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                sample_mask &= data[:, i]==v
            mask |= sample_mask

        return data[mask]


    @classmethod
    def unpack_scalar(cls, columns, data):
        """
        Given a columns object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if data.shape == (1, 1):
            return data[0, 0]
        return data


    @classmethod
    def aggregate(cls, columns, dimensions, function, **kwargs):
        reindexed = columns.reindex(dimensions)
        grouped = (cls.groupby(reindexed, dimensions, list, 'raw')
                   if len(dimensions) else [((), reindexed.data)])

        rows = []
        for k, group in grouped:
            if isinstance(function, np.ufunc):
                reduced = function.reduce(group, axis=0, **kwargs)
            else:
                reduced = function(group, axis=0, **kwargs)
            rows.append(np.concatenate([k, (reduced,) if np.isscalar(reduced) else reduced]))
        return np.atleast_2d(rows)



class DictColumns(DataColumns):
    """
    Interface for simple dictionary-based columns format. The dictionary
    keys correspond to the column (i.e dimension) names and the values
    are collections representing the values in that column.
    """

    types = (dict, OrderedDict, cyODict)

    datatype = 'dictionary'

    @classmethod
    def reshape(cls, eltype, data, kdims, vdims):
        odict_types = (OrderedDict, cyODict)
        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims

        dimensions = [d.name if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if isinstance(data, tuple):
            data = {d: v for d, v in zip(dimensions, data)}
        elif ((util.is_dataframe(data) and all(d in data for d in dimensions)) or
              (isinstance(data, NdElement) and all(d in data.dimensions() for d in dimensions))):
            data = {d: data[d] for d in dimensions}
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                if eltype._1d:
                    data = np.atleast_2d(data).T
                else:
                    data = np.column_stack([np.arange(len(data)), data])
            data = {k: data[:,i] for i,k in enumerate(dimensions)}
        elif isinstance(data, list) and np.isscalar(data[0]):
            data = {dimensions[0]: np.arange(len(data)), dimensions[1]: data}
        elif not isinstance(data, dict):
            data = {k: v for k, v in zip(dimensions, zip(*data))}
        elif isinstance(data, dict) and not all(d in data for d in dimensions):
            dict_data = zip(*((wrap_tuple(k)+wrap_tuple(v))
                              for k, v in data.items()))
            data = {k: np.array(v) for k, v in zip(dimensions, dict_data)}

        if not isinstance(data, cls.types):
            raise ValueError("DictColumns interface couldn't convert data.""")
        elif isinstance(data, dict):
            unpacked = [(d, np.array(data[d])) for d in data]
            if isinstance(data, odict_types):
                data.update(unpacked)
            else:
                data = OrderedDict([(d, np.array(data[d])) for d in dimensions])
        return data, kdims, vdims


    @classmethod
    def unpack_scalar(cls, columns, data):
        """
        Given a columns object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if len(data) != 1:
            return data
        key = list(data.keys())[0]

        if len(data[key]) == 1 and key in columns.vdims:
            return data[key][0]

    @classmethod
    def shape(cls, columns):
        return cls.length(columns), len(columns.data),

    @classmethod
    def length(cls, columns):
        return len(list(columns.data.values())[0])

    @classmethod
    def array(cls, columns, dimensions):
        if not dimensions: dimensions = columns.dimensions(label=True)
        return np.column_stack(columns.data[dim] for dim in dimensions)

    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values, vdim):
        dim = dimension.name if isinstance(dimension, Dimension) else dimension
        data = list(columns.data.items())
        if isinstance(values, basestring) or not hasattr(values, '__iter__'):
            values = np.array([values]*len(columns))
        data.insert(dim_pos, (dim, values))
        return OrderedDict(data)


    @classmethod
    def concat(cls, columns_objs):
        cast_objs = cls.cast(columns_objs)
        cols = set(tuple(c.data.keys()) for c in cast_objs)
        if len(cols) != 1:
            raise Exception("In order to concatenate, all Column objects "
                            "should have matching set of columns.")
        concatenated = OrderedDict()
        for column in cols.pop():
            concatenated[column] = np.concatenate([obj[column] for obj in cast_objs])
        return concatenated


    @classmethod
    def sort(cls, columns, by=[]):
        data = cls.array(columns, None)
        idxs = [columns.get_dimension_index(dim) for dim in by]
        sorting = np.lexsort(np.flipud(data[:, idxs].T))
        return OrderedDict([(d, v[sorting]) for d, v in columns.data.items()])

    @classmethod
    def values(cls, columns, dim):
        return np.array(columns.data.get(columns.get_dimension(dim).name))


    @classmethod
    def reindex(cls, columns, kdims, vdims):
        # DataFrame based tables don't need to be reindexed
        return OrderedDict([(d.name, columns.dimension_values(d))
                            for d in kdims+vdims])


    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        # Get dimensions information
        dimensions = [columns.get_dimension(d) for d in dimensions]
        kdims = [kdim for kdim in columns.kdims if kdim not in dimensions]
        vdims = columns.vdims

        # Update the kwargs appropriately for Element group types
        group_kwargs = {}
        group_type = dict if group_type == 'raw' else group_type
        if issubclass(group_type, Element):
            group_kwargs.update(util.get_param_values(columns))
            group_kwargs['kdims'] = kdims
        group_kwargs.update(kwargs)

        # Find all the keys along supplied dimensions
        keys = [tuple(columns.data[d.name][i] for d in dimensions)
                for i in range(len(columns))]

        # Iterate over the unique entries applying selection masks
        grouped_data = []
        for unique_key in util.unique_iterator(keys):
            mask = cls.select_mask(columns, dict(zip(dimensions, unique_key)))
            group_data = OrderedDict(((d.name, columns[d.name][mask]) for d in kdims+vdims))
            group_data = group_type(group_data, **group_kwargs)
            grouped_data.append((unique_key, group_data))

        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(grouped_data, kdims=dimensions)
        else:
            return container_type(grouped_data)


    @classmethod
    def select(cls, columns, selection_mask=None, **selection):
        if selection_mask is None:
            selection_mask = cls.select_mask(columns, selection)
        indexed = cls.indexed(columns, selection)
        data = OrderedDict((k, list(compress(v, selection_mask)))
                           for k, v in columns.data.items())
        if indexed and len(list(data.values())[0]) == 1:
            return data[columns.vdims[0].name][0]
        return data


    @classmethod
    def sample(cls, columns, samples=[]):
        mask = False
        for sample in samples:
            sample_mask = True
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                name = columns.get_dimension(i).name
                sample_mask &= (np.array(columns.data[name])==v)
            mask |= sample_mask
        return {k: np.array(col)[mask]
                for k, col in columns.data.items()}


    @classmethod
    def aggregate(cls, columns, kdims, function, **kwargs):
        kdims = [columns.get_dimension(d).name for d in kdims]
        vdims = columns.dimensions('value', True)
        groups = cls.groupby(columns, kdims, list, OrderedDict)
        aggregated = OrderedDict([(k, []) for k in kdims+vdims])

        for key, group in groups:
            key = key if isinstance(key, tuple) else (key,)
            for kdim, val in zip(kdims, key):
                aggregated[kdim].append(val)
            for vdim, arr in group.items():
                if vdim in columns.vdims:
                    if isinstance(function, np.ufunc):
                        reduced = function.reduce(arr, **kwargs)
                    else:
                        reduced = function(arr, **kwargs)
                    aggregated[vdim].append(reduced)
        return aggregated



# Register available interfaces
DataColumns.register(DictColumns)
DataColumns.register(ArrayColumns)
DataColumns.register(NdColumns)
if pd:
    DataColumns.register(DFColumns)
