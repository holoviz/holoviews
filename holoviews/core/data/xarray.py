from __future__ import absolute_import
import sys
import types

import numpy as np
import xarray as xr

try:
    import dask
    import dask.array
except ImportError:
    dask = None

from .. import util
from ..dimension import Dimension
from ..ndmapping import NdMapping, item_check, sorted_context
from ..element import Element
from .grid import GridInterface
from .interface import Interface, DataError


class XArrayInterface(GridInterface):

    types = (xr.Dataset, xr.DataArray)

    datatype = 'xarray'

    @classmethod
    def dimension_type(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
        return dataset.data[name].dtype.type


    @classmethod
    def dtype(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
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
            else:
                raise DataError("If xarray DataArray does not define a name "
                                "an explicit vdim must be supplied.", cls)
            vdims = [vdim]
            if not kdims:
                kdims = [Dimension(d) for d in data.dims[::-1]]
            data = data.to_dataset(name=vdim.name)
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
                data = {d.name: vals for d, vals in zip(kdims + vdims, data)}
            elif isinstance(data, list) and data == []:
                ndims = len(kdims)
                dimensions = [d.name if isinstance(d, Dimension) else
                              d for d in kdims + vdims]
                data = {d: np.array([]) for d in dimensions[:ndims]}
                data.update({d: np.empty((0,) * ndims) for d in dimensions[ndims:]})
            if not isinstance(data, dict):
                raise TypeError('XArrayInterface could not interpret data type')
            coords = [(kd.name, data[kd.name]) for kd in kdims][::-1]
            arrays = {}
            for vdim in vdims:
                arr = data[vdim.name]
                if not isinstance(arr, xr.DataArray):
                    arr = xr.DataArray(arr, coords=coords)
                arrays[vdim.name] = arr
            data = xr.Dataset(arrays)
        else:
            if vdims is None:
                vdims = list(data.data_vars.keys())
            if kdims is None:
                kdims = [name for name in data.indexes.keys()
                         if isinstance(data[name].data, np.ndarray)]

        kdims = [d if isinstance(d, Dimension) else Dimension(d) for d in kdims]
        not_found = [d for d in kdims if d.name not in data.coords]
        if not isinstance(data, xr.Dataset):
            raise TypeError('Data must be be an xarray Dataset type.')
        elif not_found:
            raise DataError("xarray Dataset must define coordinates "
                            "for all defined kdims, %s coordinates not found."
                            % not_found, cls)
        return data, {'kdims': kdims, 'vdims': vdims}, {}


    @classmethod
    def range(cls, dataset, dimension):
        dim = dataset.get_dimension(dimension, strict=True).name
        if dim in dataset.data:
            data = dataset.data[dim]
            dmin, dmax = data.min().data, data.max().data
            if dask and isinstance(dmin, dask.array.Array):
                dmin, dmax = dmin.compute(), dmax.compute()
            dmin = dmin if np.isscalar(dmin) else dmin.item()
            dmax = dmax if np.isscalar(dmax) else dmax.item()
            return dmin, dmax
        else:
            return np.NaN, np.NaN


    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        index_dims = [dataset.get_dimension(d, strict=True) for d in dimensions]
        element_dims = [kdim for kdim in dataset.kdims
                        if kdim not in index_dims]

        group_kwargs = {}
        if group_type != 'raw' and issubclass(group_type, Element):
            group_kwargs = dict(util.get_param_values(dataset),
                                kdims=element_dims)
        group_kwargs.update(kwargs)

        drop_dim = any(d not in group_kwargs['kdims'] for d in element_dims)

        # XArray 0.7.2 does not support multi-dimensional groupby
        # Replace custom implementation when 
        # https://github.com/pydata/xarray/pull/818 is merged.
        group_by = [d.name for d in index_dims]
        data = []
        if len(dimensions) == 1:
            for k, v in dataset.data.groupby(index_dims[0].name):
                if drop_dim:
                    v = v.to_dataframe().reset_index()
                data.append((k, group_type(v, **group_kwargs)))
        else:
            unique_iters = [cls.values(dataset, d, False) for d in group_by]
            indexes = zip(*util.cartesian_product(unique_iters))
            for k in indexes:
                sel = dataset.data.sel(**dict(zip(group_by, k)))
                if drop_dim:
                    sel = sel.to_dataframe().reset_index()
                data.append((k, group_type(sel, **group_kwargs)))

        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(data, kdims=index_dims)
        else:
            return container_type(data)


    @classmethod
    def coords(cls, dataset, dim, ordered=False, expanded=False):
        if expanded:
            return util.expand_grid_coords(dataset, dim)
        data = np.atleast_1d(dataset.data[dim].data)
        if ordered and data.shape and np.all(data[1:] < data[:-1]):
            data = data[::-1]
        return data


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True):
        dim = dataset.get_dimension(dim, strict=True)
        data = dataset.data[dim.name].data
        if dim in dataset.vdims:
            coord_dims = list(dataset.data[dim.name].dims)
            if dask and isinstance(data, dask.array.Array):
                data = data.compute()
            data = cls.canonicalize(dataset, data, coord_dims=coord_dims)
            return data.T.flatten() if flat else data
        elif expanded:
            data = cls.coords(dataset, dim.name, expanded=True)
            return data.flatten() if flat else data
        else:
            return cls.coords(dataset, dim.name, ordered=True)


    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        reduce_dims = [d.name for d in dataset.kdims if d not in dimensions]
        return dataset.data.reduce(function, dim=reduce_dims)


    @classmethod
    def unpack_scalar(cls, dataset, data):
        """
        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if (len(data.data_vars) == 1 and
            len(data[dataset.vdims[0].name].shape) == 0):
            return data[dataset.vdims[0].name].item()
        return data


    @classmethod
    def ndloc(cls, dataset, indices):
        kdims = [d.name for d in dataset.kdims[::-1]]
        adjusted_indices = []
        for kd, ind in zip(kdims, indices):
            coords = cls.coords(dataset, kd, False)
            ncoords = len(coords)
            if np.all(coords[1:] < coords[:-1]):
                if np.isscalar(ind):
                    ind = ncoords-ind-1 
                elif isinstance(ind, slice):
                    start = None if ind.stop is None else ncoords-ind.stop
                    stop = None if ind.start is None else ncoords-ind.start
                    ind = slice(start, stop, ind.step)
                elif isinstance(ind, np.ndarray) and ind.dtype.kind == 'b':
                    ind = ind[::-1]
                elif isinstance(ind, (np.ndarray, list)):
                    ind = [ncoords-i-1 for i in ind]
            if isinstance(ind, list):
                ind = np.array(ind)
            if isinstance(ind, np.ndarray) and ind.dtype.kind == 'b':
                ind = np.where(ind)[0]
            adjusted_indices.append(ind)

        isel = dict(zip(kdims, adjusted_indices))
        all_scalar = all(map(np.isscalar, indices))
        if all_scalar and len(dataset.vdims) == 1:
            return dataset.data[dataset.vdims[0].name].isel(**isel).values.item()

        # Detect if the indexing is selecting samples or slicing the array
        sampled = (all(isinstance(ind, np.ndarray) and ind.dtype.kind != 'b'
                       for ind in adjusted_indices) and len(indices) == len(kdims))
        if sampled or all_scalar:
            if all_scalar: isel = {k: [v] for k, v in isel.items()}
            return dataset.data.isel_points(**isel).to_dataframe().reset_index()
        else:
            return dataset.data.isel(**isel)


    @classmethod
    def concat(cls, dataset_objs):
        #cast_objs = cls.cast(dataset_objs)
        # Reimplement concat to automatically add dimensions
        # once multi-dimensional concat has been added to xarray.
        return xr.concat([col.data for col in dataset_objs], dim='concat_dim')

    @classmethod
    def redim(cls, dataset, dimensions):
        renames = {k: v.name for k, v in dimensions.items()}
        return dataset.data.rename(renames)

    @classmethod
    def reindex(cls, dataset, kdims=None, vdims=None):
        dropped_kdims = [kd for kd in dataset.kdims if kd not in kdims]
        constant = {}
        for kd in dropped_kdims:
            vals = cls.values(dataset, kd.name, expanded=False)
            if len(vals) == 1:
                constant[kd.name] = vals[0]
        if len(constant) == len(dropped_kdims):
            return dataset.data.sel(**constant)
        elif dropped_kdims:
            return tuple(dataset.columns(kdims+vdims).values())
        return dataset.data

    @classmethod
    def sort(cls, dataset, by=[], reverse=False):
        return dataset

    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        validated = {}
        for k, v in selection.items():
            dim = dataset.get_dimension(k, strict=True).name
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
        indexed = cls.indexed(dataset, selection)
        dropped = {d.name: np.atleast_1d(data[d.name])
                   for d in dataset.kdims
                   if not data[d.name].data.shape}
        if dropped and not indexed:
            data = data.assign_coords(**dropped)

        if (indexed and len(data.data_vars) == 1 and
            len(data[dataset.vdims[0].name].shape) == 0):
            value = data[dataset.vdims[0].name]
            if dask and isinstance(value.data, dask.array.Array):
                value = value.compute()
            return value.item()
        elif indexed:
            values = []
            for vd in dataset.vdims:
                value = data[vd.name]
                if dask and isinstance(value.data, dask.array.Array):
                    value = value.compute()
                values.append(value.item())
            return np.array(values)
        return data

    @classmethod
    def length(cls, dataset):
        return np.product([len(dataset.data[d.name]) for d in dataset.kdims])

    @classmethod
    def dframe(cls, dataset, dimensions):
        data = dataset.data.to_dataframe().reset_index()
        if dimensions:
            return data[dimensions]
        return data

    @classmethod
    def sample(cls, columns, samples=[]):
        raise NotImplementedError

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        if not vdim:
            raise Exception("Cannot add key dimension to a dense representation.")
        dim = dimension.name if isinstance(dimension, Dimension) else dimension
        arr = xr.DataArray(values, coords=dataset.data.coords, name=dim,
                           dims=dataset.data.indexes)
        return dataset.data.assign(**{dim: arr})


Interface.register(XArrayInterface)
