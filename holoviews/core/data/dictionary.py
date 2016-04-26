from collections import OrderedDict
from itertools import compress

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np

from .interface import Interface
from ..dimension import Dimension
from ..element import Element, NdElement
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
        name = dataset.get_dimension(dim).name
        return dataset.data[name].dtype.type

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
            dict_data = zip(*((util.wrap_tuple(k)+util.wrap_tuple(v))
                              for k, v in data.items()))
            data = {k: np.array(v) for k, v in zip(dimensions, dict_data)}

        if not isinstance(data, cls.types):
            raise ValueError("DictInterface interface couldn't convert data.""")
        elif isinstance(data, dict):
            unpacked = [(d, np.array(data[d])) for d in data]
            if not cls.expanded([d[1] for d in unpacked]):
                raise ValueError('DictInterface expects data to be of uniform shape.')
            if isinstance(data, odict_types):
                data.update(unpacked)
            else:
                data = OrderedDict([(d, np.array(data[d])) for d in dimensions])
        return data, {'kdims':kdims, 'vdims':vdims}, {}


    @classmethod
    def validate(cls, dataset):
        dimensions = dataset.dimensions(label=True)
        not_found = [d for d in dimensions if d not in dataset.data]
        if not_found:
            raise ValueError('Following dimensions not found in data: %s' % not_found)
        lengths = [len(dataset.data[dim]) for dim in dimensions]
        if len({l for l in lengths if l > 1}) > 1:
            raise ValueError('Length of dataset do not match')


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
    def shape(cls, dataset):
        return cls.length(dataset), len(dataset.data),

    @classmethod
    def length(cls, dataset):
        return len(list(dataset.data.values())[0])

    @classmethod
    def array(cls, dataset, dimensions):
        if not dimensions: dimensions = dataset.dimensions(label=True)
        return np.column_stack(dataset.data[dim] for dim in dimensions)

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        dim = dimension.name if isinstance(dimension, Dimension) else dimension
        data = list(dataset.data.items())
        if isinstance(values, util.basestring) or not hasattr(values, '__iter__'):
            values = np.array([values]*len(dataset))
        data.insert(dim_pos, (dim, values))
        return OrderedDict(data)


    @classmethod
    def concat(cls, dataset_objs):
        cast_objs = cls.cast(dataset_objs)
        cols = set(tuple(c.data.keys()) for c in cast_objs)
        if len(cols) != 1:
            raise Exception("In order to concatenate, all Column objects "
                            "should have matching set of dataset.")
        concatenated = OrderedDict()
        for column in cols.pop():
            concatenated[column] = np.concatenate([obj[column] for obj in cast_objs])
        return concatenated


    @classmethod
    def sort(cls, dataset, by=[]):
        if len(by) == 1:
            sorting = cls.values(dataset, by[0]).argsort()
        else:
            arrays = [dataset.dimension_values(d) for d in by]
            sorting = util.arglexsort(arrays)
        return OrderedDict([(d, v[sorting]) for d, v in dataset.data.items()])


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True):
        values = np.array(dataset.data.get(dataset.get_dimension(dim).name))
        if not expanded:
            return util.unique_array(values)
        return values


    @classmethod
    def reindex(cls, dataset, kdims, vdims):
        # DataFrame based tables don't need to be reindexed
        return OrderedDict([(d.name, dataset.dimension_values(d))
                            for d in kdims+vdims])


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
        keys = [tuple(dataset.data[d.name][i] for d in dimensions)
                for i in range(len(dataset))]

        # Iterate over the unique entries applying selection masks
        grouped_data = []
        for unique_key in util.unique_iterator(keys):
            mask = cls.select_mask(dataset, dict(zip(dimensions, unique_key)))
            group_data = OrderedDict(((d.name, dataset[d.name][mask]) for d in kdims+vdims))
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
        data = OrderedDict((k, list(compress(v, selection_mask)))
                           for k, v in dataset.data.items())
        if indexed and len(list(data.values())[0]) == 1:
            return data[dataset.vdims[0].name][0]
        return data


    @classmethod
    def sample(cls, dataset, samples=[]):
        mask = False
        for sample in samples:
            sample_mask = True
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                name = dataset.get_dimension(i).name
                sample_mask &= (np.array(dataset.data[name])==v)
            mask |= sample_mask
        return {k: np.array(col)[mask]
                for k, col in dataset.data.items()}


    @classmethod
    def aggregate(cls, dataset, kdims, function, **kwargs):
        kdims = [dataset.get_dimension(d).name for d in kdims]
        vdims = dataset.dimensions('value', True)
        groups = cls.groupby(dataset, kdims, list, OrderedDict)
        aggregated = OrderedDict([(k, []) for k in kdims+vdims])

        for key, group in groups:
            key = key if isinstance(key, tuple) else (key,)
            for kdim, val in zip(kdims, key):
                aggregated[kdim].append(val)
            for vdim, arr in group.items():
                if vdim in dataset.vdims:
                    if isinstance(function, np.ufunc):
                        reduced = function.reduce(arr, **kwargs)
                    else:
                        reduced = function(arr, **kwargs)
                    aggregated[vdim].append(reduced)
        return aggregated


Interface.register(DictInterface)
