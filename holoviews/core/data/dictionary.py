from collections import OrderedDict
from itertools import compress

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np

from .interface import Interface, DataError
from ..dimension import Dimension
from ..element import Element
from ..dimension import OrderedDict as cyODict
from ..ndmapping import NdMapping, item_check
from .. import util


class DictInterface(Interface):
    """
    Interface for simple dictionary-based dataset format. The dictionary
    keys correspond to the column (i.e dimension) names and the values
    are collections representing the values in that column.
    """

    types = (dict, OrderedDict, cyODict)

    datatype = 'dictionary'

    @classmethod
    def dimension_type(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
        values = dataset.data[name]
        return type(values) if np.isscalar(values) else values.dtype.type


    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        odict_types = (OrderedDict, cyODict)
        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims

        dimensions = [d.name if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if isinstance(data, tuple):
            data = {d: v for d, v in zip(dimensions, data)}
        elif util.is_dataframe(data) and all(d in data for d in dimensions):
            data = {d: data[d] for d in dimensions}
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                if eltype._auto_indexable_1d and len(kdims)+len(vdims)>1:
                    data = np.column_stack([np.arange(len(data)), data])
                else:
                    data = np.atleast_2d(data).T
            data = {k: data[:,i] for i,k in enumerate(dimensions)}
        elif isinstance(data, list) and data == []:
            data = OrderedDict([(d, []) for d in dimensions])
        elif isinstance(data, list) and np.isscalar(data[0]):
            data = {dimensions[0]: np.arange(len(data)), dimensions[1]: data}
        elif (isinstance(data, list) and isinstance(data[0], tuple) and len(data[0]) == 2
              and any(isinstance(v, tuple) for v in data[0])):
            dict_data = zip(*((util.wrap_tuple(k)+util.wrap_tuple(v))
                              for k, v in data))
            data = {k: np.array(v) for k, v in zip(dimensions, dict_data)}
        # Ensure that interface does not consume data of other types
        # with an iterator interface
        elif not any(isinstance(data, tuple(t for t in interface.types if t is not None))
                     for interface in cls.interfaces.values()):
            data = {k: v for k, v in zip(dimensions, zip(*data))}
        elif (isinstance(data, dict) and not any(isinstance(v, np.ndarray) for v in data.values()) and not
              any(d in data or any(d in k for k in data if isinstance(k, tuple)) for d in dimensions)):
            # For data where both keys and values are dimension values
            # e.g. {('A', 'B'): (1, 2)} (should consider deprecating)
            dict_data = sorted(data.items())
            k, v = dict_data[0]
            if len(util.wrap_tuple(k)) != len(kdims) or len(util.wrap_tuple(v)) != len(vdims):
                raise ValueError("Dictionary data not understood, should contain a column "
                                 "per dimension or a mapping between key and value dimension "
                                 "values.")
            dict_data = zip(*((util.wrap_tuple(k)+util.wrap_tuple(v))
                              for k, v in dict_data))
            data = {k: np.array(v) for k, v in zip(dimensions, dict_data)}

        if not isinstance(data, cls.types):
            raise ValueError("DictInterface interface couldn't convert data.""")
        elif isinstance(data, dict):
            unpacked = []
            for d, vals in data.items():
                if isinstance(d, tuple):
                    vals = np.asarray(vals)
                    if not vals.ndim == 2 and vals.shape[1] == len(d):
                        raise ValueError("Values for %s dimensions did not have "
                                         "the expected shape.")
                    for i, sd in enumerate(d):
                        unpacked.append((sd, vals[:, i]))
                else:
                    unpacked.append((d, vals if np.isscalar(vals) else np.asarray(vals)))
            if not cls.expanded([d[1] for d in unpacked if not np.isscalar(d[1])]):
                raise ValueError('DictInterface expects data to be of uniform shape.')
            if isinstance(data, odict_types):
                data.update(unpacked)
            else:
                data = OrderedDict(unpacked)
        return data, {'kdims':kdims, 'vdims':vdims}, {}


    @classmethod
    def validate(cls, dataset, vdims=True):
        dim_types = 'key' if vdims else 'all'
        dimensions = dataset.dimensions(dim_types, label='name')
        not_found = [d for d in dimensions if d not in dataset.data]
        if not_found:
            raise DataError('Following columns specified as dimensions '
                            'but not found in data: %s' % not_found, cls)
        lengths = [(dim, 1 if np.isscalar(dataset.data[dim]) else len(dataset.data[dim]))
                   for dim in dimensions]
        if len({l for d, l in lengths if l > 1}) > 1:
            lengths = ', '.join(['%s: %d' % l for l in sorted(lengths)])
            raise DataError('Length of columns must be equal or scalar, '
                            'columns have lengths: %s' % lengths, cls)


    @classmethod
    def unpack_scalar(cls, dataset, data):
        """
        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if len(data) != 1:
            return data
        key = list(data.keys())[0]

        if len(data[key]) == 1 and key in dataset.vdims:
            return data[key][0]

    @classmethod
    def isscalar(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
        values = dataset.data[name]
        return np.isscalar(values) or len(np.unique(values)) == 1

    @classmethod
    def shape(cls, dataset):
        return cls.length(dataset), len(dataset.data),

    @classmethod
    def length(cls, dataset):
        lengths = [len(vals) for vals in dataset.data.values() if not np.isscalar(vals)]
        return max(lengths) if lengths else 1

    @classmethod
    def array(cls, dataset, dimensions):
        if not dimensions:
            dimensions = dataset.dimensions(label='name')
        else:
            dimensions = [dataset.get_dimensions(d).name for d in dimensions]
        arrays = [dataset.data[dim.name] for dim in dimensions]
        return np.column_stack([np.full(len(dataset), arr) if np.isscalar(arr) else arr
                                for arr in arrays])

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        dim = dimension.name if isinstance(dimension, Dimension) else dimension
        data = list(dataset.data.items())
        data.insert(dim_pos, (dim, values))
        return OrderedDict(data)

    @classmethod
    def redim(cls, dataset, dimensions):
        all_dims = dataset.dimensions()
        renamed = []
        for k, v in dataset.data.items():
            if k in dimensions:
                k = dimensions[k].name
            elif k in all_dims:
                k = dataset.get_dimension(k).name
            renamed.append((k, v))
        return OrderedDict(renamed)

    @classmethod
    def concat(cls, dataset_objs):
        cast_objs = cls.cast(dataset_objs)
        cols = set(tuple(c.data.keys()) for c in cast_objs)
        if len(cols) != 1:
            raise Exception("In order to concatenate, all Dataset objects "
                            "should have matching set of columns.")
        concatenated = OrderedDict()
        for column in cols.pop():
            concatenated[column] = np.concatenate([obj[column] for obj in cast_objs])
        return concatenated


    @classmethod
    def sort(cls, dataset, by=[], reverse=False):
        by = [dataset.get_dimension(d).name for d in by]
        if len(by) == 1:
            sorting = cls.values(dataset, by[0]).argsort()
        else:
            arrays = [dataset.dimension_values(d) for d in by]
            sorting = util.arglexsort(arrays)
        return OrderedDict([(d, v if np.isscalar(v) else (v[sorting][::-1] if reverse else v[sorting]))
                            for d, v in dataset.data.items()])


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True):
        dim = dataset.get_dimension(dim).name
        values = dataset.data.get(dim)
        if np.isscalar(values):
            if not expanded:
                return np.array([values])
            values = np.full(len(dataset), values)
        else:
            if not expanded:
                return util.unique_array(values)
            values = np.array(values)
        return values


    @classmethod
    def reindex(cls, dataset, kdims, vdims):
        dimensions = [dataset.get_dimension(d).name for d in kdims+vdims]
        return OrderedDict([(d, dataset.dimension_values(d))
                            for d in dimensions])


    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        # Get dimensions information
        dimensions = [dataset.get_dimension(d) for d in dimensions]
        kdims = [kdim for kdim in dataset.kdims if kdim not in dimensions]
        vdims = dataset.vdims

        # Update the kwargs appropriately for Element group types
        group_kwargs = {}
        group_type = dict if group_type == 'raw' else group_type
        if issubclass(group_type, Element):
            group_kwargs.update(util.get_param_values(dataset))
            group_kwargs['kdims'] = kdims
        group_kwargs.update(kwargs)

        # Find all the keys along supplied dimensions
        keys = (tuple(dataset.data[d.name] if np.isscalar(dataset.data[d.name])
                      else dataset.data[d.name][i] for d in dimensions)
                for i in range(len(dataset)))

        # Iterate over the unique entries applying selection masks
        grouped_data = []
        for unique_key in util.unique_iterator(keys):
            mask = cls.select_mask(dataset, dict(zip(dimensions, unique_key)))
            group_data = OrderedDict(((d.name, dataset.data[d.name] if np.isscalar(dataset.data[d.name])
                                       else dataset.data[d.name][mask])
                                      for d in kdims+vdims))
            group_data = group_type(group_data, **group_kwargs)
            grouped_data.append((unique_key, group_data))

        if issubclass(container_type, NdMapping):
            with item_check(False):
                return container_type(grouped_data, kdims=dimensions)
        else:
            return container_type(grouped_data)


    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        if selection_mask is None:
            selection_mask = cls.select_mask(dataset, selection)
        indexed = cls.indexed(dataset, selection)
        data = OrderedDict((k, v if np.isscalar(v) else v[selection_mask])
                           for k, v in dataset.data.items())
        if indexed and len(list(data.values())[0]) == 1 and len(dataset.vdims) == 1:
            value = data[dataset.vdims[0].name]
            return value if np.isscalar(value) else value[0]
        return data


    @classmethod
    def sample(cls, dataset, samples=[]):
        mask = False
        for sample in samples:
            sample_mask = True
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                name = dataset.get_dimension(i).name
                sample_mask &= (dataset.data[name]==v)
            mask |= sample_mask
        return {k: col if np.isscalar(col) else np.array(col)[mask]
                for k, col in dataset.data.items()}


    @classmethod
    def aggregate(cls, dataset, kdims, function, **kwargs):
        kdims = [dataset.get_dimension(d, strict=True).name for d in kdims]
        vdims = dataset.dimensions('value', label='name')
        groups = cls.groupby(dataset, kdims, list, OrderedDict)
        aggregated = OrderedDict([(k, []) for k in kdims+vdims])

        for key, group in groups:
            key = key if isinstance(key, tuple) else (key,)
            for kdim, val in zip(kdims, key):
                aggregated[kdim].append(val)
            for vdim, arr in group.items():
                if vdim in dataset.vdims:
                    if np.isscalar(arr):
                        reduced = arr
                    elif isinstance(function, np.ufunc):
                        reduced = function.reduce(arr, **kwargs)
                    else:
                        reduced = function(arr, **kwargs)
                    aggregated[vdim].append(reduced)
        return aggregated


    @classmethod
    def iloc(cls, dataset, index):
        rows, cols = index
        scalar = False
        if np.isscalar(cols):
            scalar = np.isscalar(rows)
            cols = [dataset.get_dimension(cols, strict=True)]
        elif isinstance(cols, slice):
            cols = dataset.dimensions()[cols]
        else:
            cols = [dataset.get_dimension(d, strict=True) for d in cols]

        if np.isscalar(rows):
            rows = [rows]

        new_data = OrderedDict()
        for d, values in dataset.data.items():
            if d in cols:
                if np.isscalar(values):
                    new_data[d] = values
                else:
                    new_data[d] = values[rows]

        if scalar:
            arr = new_data[cols[0].name]
            return arr if np.isscalar(arr) else arr[0]
        return new_data


Interface.register(DictInterface)
