from __future__ import absolute_import
import sys
import types

import numpy as np
import xarray as xr

from .. import util
from ..dimension import Dimension
from ..ndmapping import NdMapping, item_check, sorted_context
from ..element import Element
from .grid import GridInterface
from .interface import Interface


class XArrayInterface(GridInterface):

    types = (xr.Dataset if xr else None,)

    datatype = 'xarray'

    @classmethod
    def dimension_type(cls, dataset, dim):
        name = dataset.get_dimension(dim).key
        return dataset.data[name].dtype.type


    @classmethod
    def dtype(cls, dataset, dim):
        name = dataset.get_dimension(dim).key
        return dataset.data[name].dtype


    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        element_params = eltype.params()
        kdim_param = element_params['kdims']
        vdim_param = element_params['vdims']

        if isinstance (data, xr.DataArray):
            if data.name:
                vdim = Dimension(data.name)
            elif vdims:
                vdim = vdims[0]
            elif len(vdim_param.default) == 1:
                vdim = vdim_param.default[0]
            vdims = [vdim]
            kdims = [Dimension(d) for d in data.dims[::-1]]
            data = xr.Dataset({vdim.key: data})
        elif not isinstance(data, xr.Dataset):
            if kdims is None:
                kdims = kdim_param.default
            if vdims is None:
                vdims = vdim_param.default
            kdims = [kd if isinstance(kd, Dimension) else Dimension(kd)
                     for kd in kdims]
            vdims = [vd if isinstance(vd, Dimension) else Dimension(vd)
                     for vd in vdims]
            if isinstance(data, tuple):
                data = {d.key: vals for d, vals in zip(kdims + vdims, data)}
            if not isinstance(data, dict):
                raise TypeError('XArrayInterface could not interpret data type')
            coords = [(kd.key, data[kd.key]) for kd in kdims][::-1]
            arrays = {}
            for vdim in vdims:
                arr = data[vdim.key]
                if not isinstance(arr, xr.DataArray):
                    arr = xr.DataArray(arr, coords=coords)
                arrays[vdim.key] = arr
            data = xr.Dataset(arrays)
        else:
            if vdims is None:
                vdims = list(data.data_vars.keys())
            if kdims is None:
                kdims = [name for name in data.dims
                         if isinstance(data[name].data, np.ndarray)]

        if not isinstance(data, xr.Dataset):
            raise TypeError('Data must be be an xarray Dataset type.')
        return data, {'kdims': kdims, 'vdims': vdims}, {}


    @classmethod
    def range(cls, dataset, dimension):
        dim = dataset.get_dimension(dimension).key
        if dim in dataset.data:
            data = dataset.data[dim]
            dmin, dmax = data.min().data, data.max().data
            dmin = dmin if np.isscalar(dmin) else dmin.item()
            dmax = dmax if np.isscalar(dmax) else dmax.item()
            return dmin, dmax
        else:
            return np.NaN, np.NaN


    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        index_dims = [dataset.get_dimension(d) for d in dimensions]
        element_dims = [kdim for kdim in dataset.kdims
                        if kdim not in index_dims]

        group_kwargs = {}
        if group_type != 'raw' and issubclass(group_type, Element):
            group_kwargs = dict(util.get_param_values(dataset),
                                kdims=element_dims)
        group_kwargs.update(kwargs)

        # XArray 0.7.2 does not support multi-dimensional groupby
        # Replace custom implementation when 
        # https://github.com/pydata/xarray/pull/818 is merged.
        group_by = [d.key for d in index_dims]
        if len(dimensions) == 1:
            data = [(k, group_type(v, **group_kwargs)) for k, v in
                    dataset.data.groupby(index_dims[0].key)]
        else:
            unique_iters = [cls.values(dataset, d, False) for d in group_by]
            indexes = zip(*[vals.flat for vals in util.cartesian_product(unique_iters)])
            data = [(k, group_type(dataset.data.sel(**dict(zip(group_by, k))),
                                   **group_kwargs))
                    for k in indexes]

        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(data, kdims=index_dims)
        else:
            return container_type(data)


    @classmethod
    def coords(cls, dataset, dim, ordered=False, expanded=False):
        dim = dataset.get_dimension(dim).key
        if expanded:
            return util.expand_grid_coords(dataset, dim)
        data = np.atleast_1d(dataset.data[dim].data)
        if ordered and data.shape and np.all(data[1:] < data[:-1]):
            data = data[::-1]
        return data


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True):
        dim = dataset.get_dimension(dim)
        data = dataset.data[dim.key].data
        if dim in dataset.vdims:
            coord_dims = dataset.data[dim.key].dims
            data = cls.canonicalize(dataset, data, coord_dims=coord_dims)
            return data.T.flatten() if flat else data
        elif expanded:
            data = cls.coords(dataset, dim.key, expanded=True)
            return data.flatten() if flat else data
        else:
            return cls.coords(dataset, dim.key, ordered=True)


    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        if len(dimensions) > 1:
            raise NotImplementedError('Multi-dimensional aggregation not '
                                      'supported as of xarray <=0.7.2.')
        elif not dimensions:
            return dataset.data.apply(function)
        else:
            dim = dataset.get_dimension(dimensions[0])
            return dataset.data.groupby(dim.key).apply(function)


    @classmethod
    def unpack_scalar(cls, dataset, data):
        """
        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if (len(data.data_vars) == 1 and
            len(data[dataset.vdims[0].key].shape) == 0):
            return data[dataset.vdims[0].key].item()
        return data


    @classmethod
    def concat(cls, dataset_objs):
        #cast_objs = cls.cast(dataset_objs)
        # Reimplement concat to automatically add dimensions
        # once multi-dimensional concat has been added to xarray.
        return xr.concat([col.data for col in dataset_objs], dim='concat_dim')

    @classmethod
    def redim(cls, dataset, dimensions):
        renames = {k: v.key for k, v in dimensions.items()}
        return dataset.data.rename(renames)

    @classmethod
    def reindex(cls, dataset, kdims=None, vdims=None):
        return dataset.data

    @classmethod
    def sort(cls, dataset, by=[]):
        return dataset

    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        validated = {}
        for k, v in selection.items():
            dim = dataset.get_dimension(k).key
            if isinstance(v, slice):
                v = (v.start, v.stop)
            if isinstance(v, set):
                validated[dim] = list(v)
            elif isinstance(v, tuple):
                upper = None if v[1] is None else v[1]-sys.float_info.epsilon*10
                validated[dim] = slice(v[0], upper)
            elif isinstance(v, types.FunctionType):
                validated[dim] = v(dataset[k])
            else:
                validated[dim] = v
        data = dataset.data.sel(**validated)

        # Restore constant dimensions
        dropped = {d.key: np.atleast_1d(data[d.key])
                   for d in dataset.kdims
                   if not data[d.key].data.shape}
        if dropped:
            data = data.assign_coords(**dropped)

        indexed = cls.indexed(dataset, selection)
        if (indexed and len(data.data_vars) == 1 and
            len(data[dataset.vdims[0].key].shape) == 0):
            return data[dataset.vdims[0].key].item()
        return data

    @classmethod
    def length(cls, dataset):
        return np.product(dataset[dataset.vdims[0].key].shape)
    
    @classmethod
    def dframe(cls, dataset, dimensions):
        dimensions = [dataset.get_dimension(d).key for d in dimensions]
        if dimensions:
            return dataset.reindex(columns=dimensions)
        else:
            return dataset.data.to_dataframe().reset_index(dimensions)

    @classmethod
    def sample(cls, columns, samples=[]):
        raise NotImplementedError

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        if not vdim:
            raise Exception("Cannot add key dimension to a dense representation.")
        dim = dimension.key if isinstance(dimension, Dimension) else dimension
        arr = xr.DataArray(values, coords=dataset.data.coords, name=dim,
                           dims=dataset.data.dims)
        return dataset.data.assign(**{dim: arr})


Interface.register(XArrayInterface)
