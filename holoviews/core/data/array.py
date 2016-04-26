try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np

from .interface import Interface
from ..dimension import Dimension
from ..element import Element
from ..ndmapping import NdMapping, item_check
from .. import util


class ArrayInterface(Interface):

    types = (np.ndarray,)

    datatype = 'array'

    @classmethod
    def dimension_type(cls, dataset, dim):
        return dataset.data.dtype.type

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims

        dimensions = [d.name if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if ((isinstance(data, dict) or util.is_dataframe(data)) and
            all(d in data for d in dimensions)):
            dataset = [data[d] for d in dimensions]
            data = np.column_stack(dataset)
        elif isinstance(data, dict) and not all(d in data for d in dimensions):
            dataset = zip(*((util.wrap_tuple(k)+util.wrap_tuple(v))
                            for k, v in data.items()))
            data = np.column_stack(dataset)
        elif isinstance(data, tuple):
            data = [d if isinstance(d, np.ndarray) else np.array(d) for d in data]
            if cls.expanded(data):
                data = np.column_stack(data)
            else:
                raise ValueError('ArrayInterface expects data to be of uniform shape.')
        elif not isinstance(data, np.ndarray):
            data = np.array([], ndmin=2).T if data is None else list(data)
            try:
                data = np.array(data)
            except:
                data = None

        if data is None or data.ndim > 2 or data.dtype.kind in ['S', 'U', 'O']:
            raise ValueError("ArrayInterface interface could not handle input type.")
        elif data.ndim == 1:
            if eltype._1d:
                data = np.atleast_2d(data).T
            else:
                data = np.column_stack([np.arange(len(data)), data])

        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims
        return data, {'kdims':kdims, 'vdims':vdims}, {}

    @classmethod
    def validate(cls, dataset):
        ndims = len(dataset.dimensions())
        ncols = dataset.data.shape[1] if dataset.data.ndim > 1 else 1
        if ncols < ndims:
            raise ValueError("Supplied data does not match specified "
                             "dimensions, expected at least %s dataset." % ndims)

    @classmethod
    def array(cls, dataset, dimensions):
        if dimensions:
            return Element.dframe(dataset, dimensions)
        else:
            return dataset.data


    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        data = dataset.data.copy()
        return np.insert(data, dim_pos, values, axis=1)


    @classmethod
    def concat(cls, dataset_objs):
        cast_objs = cls.cast(dataset_objs)
        return np.concatenate([col.data for col in cast_objs])


    @classmethod
    def sort(cls, dataset, by=[]):
        data = dataset.data
        if len(by) == 1:
            sorting = cls.values(dataset, by[0]).argsort()
        else:
            dtypes = (dataset.data.dtype,)*dataset.data.shape[1]
            sort_fields = tuple('f%s' % dataset.get_dimension_index(d) for d in by)
            sorting = dataset.data.T.view(dtypes).argsort(order=sort_fields)
        return data[sorting]


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True):
        data = dataset.data
        dim_idx = dataset.get_dimension_index(dim)
        if data.ndim == 1:
            data = np.atleast_2d(data).T
        values = data[:, dim_idx]
        if not expanded:
            return util.unique_array(values)
        return values


    @classmethod
    def reindex(cls, dataset, kdims=None, vdims=None):
        # DataFrame based tables don't need to be reindexed
        dims = kdims + vdims
        data = [dataset.dimension_values(d) for d in dims]
        return np.column_stack(data)


    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        data = dataset.data

        # Get dimension objects, labels, indexes and data
        dimensions = [dataset.get_dimension(d) for d in dimensions]
        dim_idxs = [dataset.get_dimension_index(d) for d in dimensions]
        ndims = len(dimensions)
        kdims = [kdim for kdim in dataset.kdims
                 if kdim not in dimensions]
        vdims = dataset.vdims

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
            group_kwargs.update(util.get_param_values(dataset))
            group_kwargs['kdims'] = kdims
        group_kwargs.update(kwargs)

        # Iterate over the unique entries building masks
        # to apply the group selection
        grouped_data = []
        for group in unique_indices:
            mask = np.logical_and.reduce([data[:, d_idx] == group[i]
                                          for i, d_idx in enumerate(dim_idxs)])
            group_data = data[mask, ndims:]
            if not group_type == 'raw':
                if issubclass(group_type, dict):
                    group_data = {d.name: group_data[:, i] for i, d in
                                  enumerate(kdims+vdims)}
                else:
                    group_data = group_type(group_data, **group_kwargs)
            grouped_data.append((tuple(group), group_data))

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
        data = np.atleast_2d(dataset.data[selection_mask, :])
        if len(data) == 1 and indexed:
            data = data[0, dataset.ndims]
        return data


    @classmethod
    def sample(cls, dataset, samples=[]):
        data = dataset.data
        mask = False
        for sample in samples:
            sample_mask = True
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                sample_mask &= data[:, i]==v
            mask |= sample_mask

        return data[mask]


    @classmethod
    def unpack_scalar(cls, dataset, data):
        """
        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if data.shape == (1, 1):
            return data[0, 0]
        return data


    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        reindexed = dataset.reindex(dimensions)
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


Interface.register(ArrayInterface)
