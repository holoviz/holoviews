from __future__ import absolute_import
import sys
import types
from collections import OrderedDict

import numpy as np

from .. import util
from ..dimension import Dimension, asdim, dimension_name
from ..ndmapping import NdMapping, item_check, sorted_context
from ..element import Element
from .grid import GridInterface
from .interface import Interface, DataError, dask_array_module


class XArrayInterface(GridInterface):

    types = ()

    datatype = 'xarray'

    @classmethod
    def loaded(cls):
        return 'xarray' in sys.modules

    @classmethod
    def applies(cls, obj):
        if not cls.loaded():
            return False
        import xarray as xr
        return isinstance(obj, (xr.Dataset, xr.DataArray))

    @classmethod
    def dimension_type(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
        return dataset.data[name].dtype.type


    @classmethod
    def dtype(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
        return dataset.data[name].dtype


    @classmethod
    def shape(cls, dataset, gridded=False):
        array = dataset.data[dataset.vdims[0].name]
        if not any(cls.irregular(dataset, kd) for kd in dataset.kdims):
            names = [kd.name for kd in dataset.kdims
                     if kd.name in array.dims][::-1]
            if not all(d in names for d in array.dims):
                array = np.squeeze(array)
            try:
                array = array.transpose(*names, transpose_coords=False)
            except:
                array = array.transpose(*names) # Handle old xarray
        shape = array.shape
        if gridded:
            return shape
        else:
            return (np.product(shape, dtype=np.intp), len(dataset.dimensions()))


    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        import xarray as xr
        element_params = eltype.param.objects()
        kdim_param = element_params['kdims']
        vdim_param = element_params['vdims']

        def retrieve_unit_and_label(dim):
            if isinstance(dim, Dimension):
                return dim
            dim = asdim(dim)
            coord = data[dim.name]
            unit = coord.attrs.get('units') if dim.unit is None else dim.unit
            if 'long_name' in coord.attrs:
                spec = (dim.name, coord.attrs['long_name'])
            else:
                spec = (dim.name, dim.label)
            return dim.clone(spec, unit=unit)

        if isinstance(data, xr.DataArray):
            if vdims:
                vdim = vdims[0]
            elif data.name:
                vdim = Dimension(data.name)
                vdim.unit = data.attrs.get('units')
                label = data.attrs.get('long_name')
                if label is not None:
                    vdim.label = label
            elif len(vdim_param.default) == 1:
                vdim = vdim_param.default[0]
                if vdim.name in data.dims:
                    raise DataError("xarray DataArray does not define a name, "
                                    "and the default of '%s' clashes with a "
                                    "coordinate dimension. Give the DataArray "
                                    "a name or supply an explicit value dimension."
                                    % vdim.name, cls)
            else:
                raise DataError("xarray DataArray does not define a name "
                                "and %s does not define a default value "
                                "dimension. Give the DataArray a name or "
                                "supply an explicit vdim." % eltype.__name__,
                                cls)
            vdims = [vdim]
            data = data.to_dataset(name=vdim.name)

        if not isinstance(data, xr.Dataset):
            if kdims is None:
                kdims = kdim_param.default
            if vdims is None:
                vdims = vdim_param.default
            kdims = [asdim(kd) for kd in kdims]
            vdims = [asdim(vd) for vd in vdims]
            if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == len(kdims+vdims):
                data = tuple(data)
            if isinstance(data, tuple):
                data = {d.name: vals for d, vals in zip(kdims + vdims, data)}
            elif isinstance(data, list) and data == []:
                ndims = len(kdims)
                dimensions = [d.name for d in kdims + vdims]
                data = {d: np.array([]) for d in dimensions[:ndims]}
                data.update({d: np.empty((0,) * ndims) for d in dimensions[ndims:]})
            if not isinstance(data, dict):
                raise TypeError('XArrayInterface could not interpret data type')
            data = {d: np.asarray(values) if d in kdims else values
                    for d, values in data.items()}
            coord_dims = [data[kd.name].ndim for kd in kdims]
            dims = tuple('dim_%d' % i for i in range(max(coord_dims)))[::-1]
            coords = OrderedDict()
            for kd in kdims:
                coord_vals = data[kd.name]
                if coord_vals.ndim > 1:
                    coord = (dims[:coord_vals.ndim], coord_vals)
                else:
                    coord = coord_vals
                coords[kd.name] = coord
            xr_kwargs = {'dims': dims if max(coord_dims) > 1 else list(coords)[::-1]}
            arrays = {}
            for vdim in vdims:
                arr = data[vdim.name]
                if not isinstance(arr, xr.DataArray):
                    arr = xr.DataArray(arr, coords=coords, **xr_kwargs)
                arrays[vdim.name] = arr
            data = xr.Dataset(arrays)
        else:
            if not data.coords:
                data = data.assign_coords(**{k: range(v) for k, v in data.dims.items()})
            if vdims is None:
                vdims = list(data.data_vars)
            if kdims is None:
                xrdims = list(data.dims)
                xrcoords = list(data.coords)
                kdims = [name for name in data.indexes.keys()
                         if isinstance(data[name].data, np.ndarray)]
                kdims = sorted(kdims, key=lambda x: (xrcoords.index(x) if x in xrcoords else float('inf'), x))
                if set(xrdims) != set(kdims):
                    virtual_dims = [xd for xd in xrdims if xd not in kdims]
                    for c in data.coords:
                        if c not in kdims and set(data[c].dims) == set(virtual_dims):
                            kdims.append(c)
            kdims = [retrieve_unit_and_label(kd) for kd in kdims]
            vdims = [retrieve_unit_and_label(vd) for vd in vdims]

        not_found = []
        for d in kdims:
            if not any(d.name == k or (isinstance(v, xr.DataArray) and d.name in v.dims)
                       for k, v in data.coords.items()):
                not_found.append(d)
        if not isinstance(data, xr.Dataset):
            raise TypeError('Data must be be an xarray Dataset type.')
        elif not_found:
            raise DataError("xarray Dataset must define coordinates "
                            "for all defined kdims, %s coordinates not found."
                            % not_found, cls)

        return data, {'kdims': kdims, 'vdims': vdims}, {}


    @classmethod
    def validate(cls, dataset, vdims=True):
        Interface.validate(dataset, vdims)
        # Check whether irregular (i.e. multi-dimensional) coordinate
        # array dimensionality matches
        irregular = []
        for kd in dataset.kdims:
            if cls.irregular(dataset, kd):
                irregular.append((kd, dataset.data[kd.name].dims))
        if irregular:
            nonmatching = ['%s: %s' % (kd, dims) for kd, dims in irregular[1:]
                           if set(dims) != set(irregular[0][1])]
            if nonmatching:
                nonmatching = ['%s: %s' % irregular[0]] + nonmatching
                raise DataError("The dimensions of coordinate arrays "
                                "on irregular data must match. The "
                                "following kdims were found to have "
                                "non-matching array dimensions:\n\n%s"
                                % ('\n'.join(nonmatching)), cls)


    @classmethod
    def range(cls, dataset, dimension):
        dim = dataset.get_dimension(dimension, strict=True).name
        if dataset._binned and dimension in dataset.kdims:
            data = cls.coords(dataset, dim, edges=True)
            if data.dtype.kind == 'M':
                dmin, dmax = data.min(), data.max()
            else:
                dmin, dmax = np.nanmin(data), np.nanmax(data)
        else:
            data = dataset.data[dim]
            if len(data):
                dmin, dmax = data.min().data, data.max().data
            else:
                dmin, dmax = np.NaN, np.NaN

        da = dask_array_module()
        if da and isinstance(dmin, da.Array):
            dmin, dmax = da.compute(dmin, dmax)
        dmin = dmin if np.isscalar(dmin) or isinstance(dmin, util.datetime_types) else dmin.item()
        dmax = dmax if np.isscalar(dmax) or isinstance(dmax, util.datetime_types) else dmax.item()
        return dmin, dmax


    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        index_dims = [dataset.get_dimension(d, strict=True) for d in dimensions]
        element_dims = [kdim for kdim in dataset.kdims
                        if kdim not in index_dims]

        invalid = [d for d in index_dims if dataset.data[d.name].ndim > 1]
        if invalid:
            if len(invalid) == 1: invalid = "'%s'" % invalid[0]
            raise ValueError("Cannot groupby irregularly sampled dimension(s) %s."
                             % invalid)

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
    def coords(cls, dataset, dimension, ordered=False, expanded=False, edges=False):
        import xarray as xr
        dim = dataset.get_dimension(dimension)
        dim = dimension if dim is None else dim.name
        irregular = cls.irregular(dataset, dim)
        if irregular or expanded:
            if irregular:
                data = dataset.data[dim]
            else:
                data = util.expand_grid_coords(dataset, dim)
            if edges:
                data = cls._infer_interval_breaks(data, axis=1)
                data = cls._infer_interval_breaks(data, axis=0)

            return data.values if isinstance(data, xr.DataArray) else data

        data = np.atleast_1d(dataset.data[dim].data)
        if ordered and data.shape and np.all(data[1:] < data[:-1]):
            data = data[::-1]
        shape = cls.shape(dataset, True)

        if dim in dataset.kdims:
            idx = dataset.get_dimension_index(dim)
            isedges = (dim in dataset.kdims and len(shape) == dataset.ndims
                       and len(data) == (shape[dataset.ndims-idx-1]+1))
        else:
            isedges = False
        if edges and not isedges:
            data = cls._infer_interval_breaks(data)
        elif not edges and isedges:
            data = np.convolve(data, [0.5, 0.5], 'valid')

        return data.values if isinstance(data, xr.DataArray) else data


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True, compute=True, keep_index=False):
        dim = dataset.get_dimension(dim, strict=True)
        data = dataset.data[dim.name].data
        irregular = cls.irregular(dataset, dim) if dim in dataset.kdims else False
        irregular_kdims = [d for d in dataset.kdims if cls.irregular(dataset, d)]
        if irregular_kdims:
            virtual_coords = list(dataset.data[irregular_kdims[0].name].coords.dims)
        else:
            virtual_coords = []
        if dim in dataset.vdims or irregular:
            data_coords = list(dataset.data[dim.name].dims)
            da = dask_array_module()
            if compute and da and isinstance(data, da.Array):
                data = data.compute()
            data = cls.canonicalize(dataset, data, data_coords=data_coords,
                                    virtual_coords=virtual_coords)
            return data.T.flatten() if flat else data
        elif expanded:
            data = cls.coords(dataset, dim.name, expanded=True)
            return data.T.flatten() if flat else data
        else:
            return cls.coords(dataset, dim.name, ordered=True)


    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        reduce_dims = [d.name for d in dataset.kdims if d not in dimensions]
        return dataset.data.reduce(function, dim=reduce_dims), []


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
        kdims = [d for d in dataset.kdims[::-1]]
        adjusted_indices = []
        slice_dims = []
        for kd, ind in zip(kdims, indices):
            if cls.irregular(dataset, kd):
                coords = [c for c in dataset.data.coords if c not in dataset.data.dims]
                dim = dataset.data[kd.name].dims[coords.index(kd.name)]
                shape = dataset.data[kd.name].shape[coords.index(kd.name)]
                coords = np.arange(shape)
            else:
                coords = cls.coords(dataset, kd, False)
                dim = kd.name
            slice_dims.append(dim)
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

        isel = dict(zip(slice_dims, adjusted_indices))
        all_scalar = all(map(np.isscalar, indices))
        if all_scalar and len(indices) == len(kdims) and len(dataset.vdims) == 1:
            return dataset.data[dataset.vdims[0].name].isel(**isel).values.item()

        # Detect if the indexing is selecting samples or slicing the array
        sampled = (all(isinstance(ind, np.ndarray) and ind.dtype.kind != 'b'
                       for ind in adjusted_indices) and len(indices) == len(kdims))
        if sampled or (all_scalar and len(indices) == len(kdims)):
            if all_scalar: isel = {k: [v] for k, v in isel.items()}
            return dataset.data.isel_points(**isel).to_dataframe().reset_index()
        else:
            return dataset.data.isel(**isel)

    @classmethod
    def concat_dim(cls, datasets, dim, vdims):
        import xarray as xr
        return xr.concat([ds.assign_coords(**{dim.name: c}) for c, ds in datasets.items()],
                         dim=dim.name)

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
            return dataset.data.sel(**{k: v for k, v in constant.items()
                                       if k in dataset.data.dims})
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
            dim = dataset.get_dimension(k, strict=True)
            if cls.irregular(dataset, dim):
                return GridInterface.select(dataset, selection_mask, **selection)
            dim = dim.name
            if isinstance(v, slice):
                v = (v.start, v.stop)
            if isinstance(v, set):
                validated[dim] = list(v)
            elif isinstance(v, tuple):
                dim_vals = dataset.data[k].values
                upper = None if v[1] is None else v[1]-sys.float_info.epsilon*10
                v = v[0], upper
                if dim_vals.dtype.kind not in 'OSU' and np.all(dim_vals[1:] < dim_vals[:-1]):
                    # If coordinates are inverted invert slice
                    v = v[::-1]
                validated[dim] = slice(*v)
            elif isinstance(v, types.FunctionType):
                validated[dim] = v(dataset[k])
            else:
                validated[dim] = v
        data = dataset.data.sel(**validated)

        # Restore constant dimensions
        indexed = cls.indexed(dataset, selection)
        dropped = OrderedDict((d.name, np.atleast_1d(data[d.name]))
                   for d in dataset.kdims
                   if not data[d.name].data.shape)
        if dropped and not indexed:
            data = data.expand_dims(dropped)
            # see https://github.com/pydata/xarray/issues/2891
            # since we only exapanded on dimnesions of size 1
            # we can monkeypatch the dataarray back to writeable.
            for d in data.values():
                if hasattr(d.data, 'flags'):
                    d.data.flags.writeable = True

        da = dask_array_module()
        if (indexed and len(data.data_vars) == 1 and
            len(data[dataset.vdims[0].name].shape) == 0):
            value = data[dataset.vdims[0].name]
            if da and isinstance(value.data, da.Array):
                value = value.compute()
            return value.item()
        elif indexed:
            values = []
            for vd in dataset.vdims:
                value = data[vd.name]
                if da and isinstance(value.data, da.Array):
                    value = value.compute()
                values.append(value.item())
            return np.array(values)
        return data

    @classmethod
    def length(cls, dataset):
        return np.product([len(dataset.data[d.name]) for d in dataset.kdims], dtype=np.intp)

    @classmethod
    def dframe(cls, dataset, dimensions):
        data = dataset.data.to_dataframe().reset_index()
        if dimensions:
            return data[dimensions]
        return data

    @classmethod
    def sample(cls, dataset, samples=[]):
        raise NotImplementedError

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        import xarray as xr
        if not vdim:
            raise Exception("Cannot add key dimension to a dense representation.")
        dim = dimension_name(dimension)
        coords = {d.name: cls.coords(dataset, d.name) for d in dataset.kdims}
        arr = xr.DataArray(values, coords=coords, name=dim,
                           dims=tuple(d.name for d in dataset.kdims[::-1]))
        return dataset.data.assign(**{dim: arr})


Interface.register(XArrayInterface)
