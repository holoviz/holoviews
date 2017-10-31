from collections import OrderedDict, defaultdict, Iterable

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np

from .dictionary import DictInterface
from .interface import Interface, DataError
from ..dimension import Dimension
from ..element import Element
from ..dimension import OrderedDict as cyODict
from ..ndmapping import NdMapping, item_check
from .. import util



class GridInterface(DictInterface):
    """
    Interface for simple dictionary-based dataset format using a
    compressed representation that uses the cartesian product between
    key dimensions. As with DictInterface, the dictionary keys correspond
    to the column (i.e dimension) names and the values are NumPy arrays
    representing the values in that column.

    To use this compressed format, the key dimensions must be orthogonal
    to one another with each key dimension specifying an axis of the
    multidimensional space occupied by the value dimension data. For
    instance, given an temperature recordings sampled regularly across
    the earth surface, a list of N unique latitudes and M unique
    longitudes can specify the position of NxM temperature samples.
    """

    types = (dict, OrderedDict, cyODict)

    datatype = 'grid'

    gridded = True

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims

        if not vdims:
            raise ValueError('GridInterface interface requires at least '
                             'one value dimension.')

        ndims = len(kdims)
        dimensions = [d.name if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if isinstance(data, tuple):
            data = {d: v for d, v in zip(dimensions, data)}
        elif isinstance(data, list) and data == []:
            data = {d: np.array([]) for d in dimensions[:ndims]}
            data.update({d: np.empty((0,) * ndims) for d in dimensions[ndims:]})
        elif not isinstance(data, dict):
            raise TypeError('GridInterface must be instantiated as a '
                            'dictionary or tuple')

        for dim in kdims+vdims:
            name = dim.name if isinstance(dim, Dimension) else dim
            if name not in data:
                raise ValueError("Values for dimension %s not found" % dim)
            if not isinstance(data[name], np.ndarray):
                data[name] = np.array(data[name])

        kdim_names = [d.name if isinstance(d, Dimension) else d for d in kdims]
        vdim_names = [d.name if isinstance(d, Dimension) else d for d in vdims]
        expected = tuple([len(data[kd]) for kd in kdim_names])
        for vdim in vdim_names:
            shape = data[vdim].shape
            error = DataError if len(shape) > 1 else ValueError
            if shape != expected[::-1] and not (not expected and shape == (1,)):
                raise error('Key dimension values and value array %s '
                            'shapes do not match. Expected shape %s, '
                            'actual shape: %s' % (vdim, expected[::-1], shape), cls)
        return data, {'kdims':kdims, 'vdims':vdims}, {}


    @classmethod
    def isscalar(cls, dataset, dim):
        return np.unique(cls.values(dataset, dim, expanded=False)) == 1


    @classmethod
    def validate(cls, dataset, vdims=True):
        Interface.validate(dataset, vdims)


    @classmethod
    def dimension_type(cls, dataset, dim):
        if dim in dataset.dimensions():
            arr = cls.values(dataset, dim, False, False)
        else:
            return None
        return arr.dtype.type


    @classmethod
    def shape(cls, dataset, gridded=False):
        if gridded:
            return dataset.data[dataset.vdims[0].name].shape
        else:
            return (cls.length(dataset), len(dataset.dimensions()))


    @classmethod
    def length(cls, dataset):
        return np.product([len(dataset.data[d.name]) for d in dataset.kdims])


    @classmethod
    def coords(cls, dataset, dim, ordered=False, expanded=False):
        """
        Returns the coordinates along a dimension.  Ordered ensures
        coordinates are in ascending order and expanded creates
        ND-array matching the dimensionality of the dataset.
        """
        dim = dataset.get_dimension(dim, strict=True)
        if expanded:
            return util.expand_grid_coords(dataset, dim)
        data = dataset.data[dim.name]
        if ordered and np.all(data[1:] < data[:-1]):
            data = data[::-1]
        return data


    @classmethod
    def canonicalize(cls, dataset, data, coord_dims=None):
        """
        Canonicalize takes an array of values as input and
        reorients and transposes it to match the canonical
        format expected by plotting functions. In addition
        to the dataset and the particular array to apply
        transforms to a list of coord_dims may be supplied
        in case the array indexing does not match the key
        dimensions of the dataset.
        """
        if coord_dims is None:
            coord_dims = dataset.dimensions('key', label='name')[::-1]

        # Reorient data
        invert = False
        slices = []
        for d in coord_dims:
            coords = cls.coords(dataset, d)
            if np.all(coords[1:] < coords[:-1]):
                slices.append(slice(None, None, -1))
                invert = True
            else:
                slices.append(slice(None))
        data = data[slices] if invert else data

        # Transpose data
        dims = [name for name in coord_dims
                if isinstance(cls.coords(dataset, name), np.ndarray)]
        dropped = [dims.index(d) for d in dims if d not in dataset.kdims]
        inds = [dims.index(kd.name)for kd in dataset.kdims]
        inds = [i - sum([1 for d in dropped if i>=d]) for i in inds]
        if dropped:
            data = data.squeeze(axis=tuple(dropped))
        if inds:
            data = data.transpose(inds[::-1])

        # Allow lower dimensional views into data
        if len(dataset.kdims) < 2:
            data = data.flatten()
        return data


    @classmethod
    def invert_index(cls, index, length):
        if np.isscalar(index):
            return length - index
        elif isinstance(index, slice):
            start, stop = index.start, index.stop
            new_start, new_stop = None, None
            if start is not None:
                new_stop = length - start
            if stop is not None:
                new_start = length - stop
            return slice(new_start-1, new_stop-1)
        elif isinstance(index, Iterable):
            new_index = []
            for ind in index:
                new_index.append(length-ind)
        return new_index


    @classmethod
    def ndloc(cls, dataset, indices):
        selected = {}
        adjusted_inds = []
        all_scalar = True
        for kd, ind in zip(dataset.kdims[::-1], indices):
            coords = cls.coords(dataset, kd.name, True)
            if np.isscalar(ind):
                ind = [ind]
            else:
                all_scalar = False
            selected[kd.name] = coords[ind]
            adjusted_inds.append(ind)
        for kd in dataset.kdims:
            if kd.name not in selected:
                coords = cls.coords(dataset, kd.name)
                selected[kd.name] = coords
                all_scalar = False
        for vd in dataset.vdims:
            arr = dataset.dimension_values(vd, flat=False)
            if all_scalar and len(dataset.vdims) == 1:
                return arr[tuple(ind[0] for ind in adjusted_inds)]
            selected[vd.name] = arr[tuple(adjusted_inds)]
        return tuple(selected[d.name] for d in dataset.dimensions())


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True):
        dim = dataset.get_dimension(dim, strict=True)
        if dim in dataset.vdims:
            data = dataset.data.get(dim.name)
            data = cls.canonicalize(dataset, data)
            return data.T.flatten() if flat else data
        elif expanded:
            data = cls.coords(dataset, dim.name, expanded=True)
            return data.flatten() if flat else data
        else:
            return cls.coords(dataset, dim.name, ordered=True)


    @classmethod
    def groupby(cls, dataset, dim_names, container_type, group_type, **kwargs):
        # Get dimensions information
        dimensions = [dataset.get_dimension(d, strict=True) for d in dim_names]
        kdims = [kdim for kdim in dataset.kdims if kdim not in dimensions]

        # Update the kwargs appropriately for Element group types
        group_kwargs = {}
        group_type = dict if group_type == 'raw' else group_type
        if issubclass(group_type, Element):
            group_kwargs.update(util.get_param_values(dataset))
            group_kwargs['kdims'] = kdims
        group_kwargs.update(kwargs)

        drop_dim = any(d not in group_kwargs['kdims'] for d in kdims)

        # Find all the keys along supplied dimensions
        keys = [dataset.data[d.name] for d in dimensions]

        # Iterate over the unique entries applying selection masks
        grouped_data = []
        for unique_key in zip(*util.cartesian_product(keys)):
            select = dict(zip(dim_names, unique_key))
            if drop_dim:
                group_data = dataset.select(**select)
                group_data = group_data if np.isscalar(group_data) else group_data.columns()
            else:
                group_data = cls.select(dataset, **select)
            if np.isscalar(group_data):
                group_data = {dataset.vdims[0].name: np.atleast_1d(group_data)}
                for dim, v in zip(dim_names, unique_key):
                    group_data[dim] = np.atleast_1d(v)
            elif not drop_dim:
                for vdim in dataset.vdims:
                    group_data[vdim.name] = np.squeeze(group_data[vdim.name])
            group_data = group_type(group_data, **group_kwargs)
            grouped_data.append((tuple(unique_key), group_data))

        if issubclass(container_type, NdMapping):
            with item_check(False):
                return container_type(grouped_data, kdims=dimensions)
        else:
            return container_type(grouped_data)


    @classmethod
    def key_select_mask(cls, dataset, values, ind):
        if isinstance(ind, tuple):
            ind = slice(*ind)
        if isinstance(ind, np.ndarray):
            mask = ind
        elif isinstance(ind, slice):
            mask = True
            if ind.start is not None:
                mask &= ind.start <= values
            if ind.stop is not None:
                mask &= values < ind.stop
            # Expand empty mask
            if mask is True:
                mask = np.ones(values.shape, dtype=np.bool)
        elif isinstance(ind, (set, list)):
            iter_slcs = []
            for ik in ind:
                iter_slcs.append(values == ik)
            mask = np.logical_or.reduce(iter_slcs)
        elif callable(ind):
            mask = ind(values)
        elif ind is None:
            mask = None
        else:
            index_mask = values == ind
            if dataset.ndims == 1 and np.sum(index_mask) == 0:
                data_index = np.argmin(np.abs(values - ind))
                mask = np.zeros(len(dataset), dtype=np.bool)
                mask[data_index] = True
            else:
                mask = index_mask
        return mask


    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        dimensions = dataset.kdims
        val_dims = [vdim for vdim in dataset.vdims if vdim in selection]
        if val_dims:
            raise IndexError('Cannot slice value dimensions in compressed format, '
                             'convert to expanded format before slicing.')

        indexed = cls.indexed(dataset, selection)
        selection = [(d, selection.get(d.name, selection.get(d.label)))
                      for d in dimensions]
        data = {}
        value_select = []
        for dim, ind in selection:
            values = cls.values(dataset, dim, False)
            mask = cls.key_select_mask(dataset, values, ind)
            if mask is None:
                mask = np.ones(values.shape, dtype=bool)
            else:
                values = values[mask]
            value_select.append(mask)
            data[dim.name] = np.array([values]) if np.isscalar(values) else values
        int_inds = [np.argwhere(v) for v in value_select][::-1]
        index = np.ix_(*[np.atleast_1d(np.squeeze(ind)) if ind.ndim > 1 else np.atleast_1d(ind)
                         for ind in int_inds])
        for vdim in dataset.vdims:
            data[vdim.name] = dataset.data[vdim.name][index]

        if indexed:
            if len(dataset.vdims) == 1:
                arr = np.squeeze(data[dataset.vdims[0].name])
                return arr if np.isscalar(arr) else arr[()]
            else:
                return np.array([np.squeeze(data[vd.name])
                                 for vd in dataset.vdims])
        return data


    @classmethod
    def sample(cls, dataset, samples=[]):
        """
        Samples the gridded data into dataset of samples.
        """
        ndims = dataset.ndims
        dimensions = dataset.dimensions(label='name')
        arrays = [dataset.data[vdim.name] for vdim in dataset.vdims]
        data = defaultdict(list)

        for sample in samples:
            if np.isscalar(sample): sample = [sample]
            if len(sample) != ndims:
                sample = [sample[i] if i < len(sample) else None
                          for i in range(ndims)]
            sampled, int_inds = [], []
            for d, ind in zip(dimensions, sample):
                cdata = dataset.data[d]
                mask = cls.key_select_mask(dataset, cdata, ind)
                inds = np.arange(len(cdata)) if mask is None else np.argwhere(mask)
                int_inds.append(inds)
                sampled.append(cdata[mask])
            for d, arr in zip(dimensions, np.meshgrid(*sampled)):
                data[d].append(arr)
            for vdim, array in zip(dataset.vdims, arrays):
                flat_index = np.ravel_multi_index(tuple(int_inds)[::-1], array.shape)
                data[vdim.name].append(array.flat[flat_index])
        concatenated = {d: np.concatenate(arrays).flatten() for d, arrays in data.items()}
        return concatenated


    @classmethod
    def aggregate(cls, dataset, kdims, function, **kwargs):
        kdims = [kd.name if isinstance(kd, Dimension) else kd for kd in kdims]
        data = {kdim: dataset.data[kdim] for kdim in kdims}
        axes = tuple(dataset.ndims-dataset.get_dimension_index(kdim)-1
                     for kdim in dataset.kdims if kdim not in kdims)
        for vdim in dataset.vdims:
            data[vdim.name] = np.atleast_1d(function(dataset.data[vdim.name],
                                                      axis=axes, **kwargs))

        return data


    @classmethod
    def reindex(cls, dataset, kdims, vdims):
        dropped_kdims = [kd for kd in dataset.kdims if kd not in kdims]
        dropped_vdims = ([vdim for vdim in dataset.vdims
                          if vdim not in vdims] if vdims else [])
        constant = {}
        for kd in dropped_kdims:
            vals = cls.values(dataset, kd.name, expanded=False)
            if len(vals) == 1:
                constant[kd.name] = vals[0]
        data = {k: values for k, values in dataset.data.items()
                if k not in dropped_kdims+dropped_vdims}

        if len(constant) == len(dropped_kdims):
            joined_dims = kdims+dropped_kdims
            axes = tuple(dataset.ndims-dataset.kdims.index(d)-1
                         for d in joined_dims)
            dropped_axes = tuple(dataset.ndims-joined_dims.index(d)-1
                                 for d in dropped_kdims)
            for vdim in vdims:
                vdata = data[vdim.name]
                if len(axes) > 1:
                    vdata = vdata.transpose(axes[::-1])
                if dropped_axes:
                    vdata = vdata.squeeze(axis=dropped_axes)
                data[vdim.name] = vdata
            return data
        elif dropped_kdims:
            return tuple(dataset.columns(kdims+vdims).values())
        return data


    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        if not vdim:
            raise Exception("Cannot add key dimension to a dense representation.")
        dim = dimension.name if isinstance(dimension, Dimension) else dimension
        return dict(dataset.data, **{dim: values})


    @classmethod
    def sort(cls, dataset, by=[], reverse=False):
        if not by or by in [dataset.kdims, dataset.dimensions()]:
            return dataset.data
        else:
            raise Exception('Compressed format cannot be sorted, either instantiate '
                            'in the desired order or use the expanded format.')

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

        new_data = []
        for d in cols:
            new_data.append(dataset.dimension_values(d)[rows])

        if scalar:
            return new_data[0][0]
        return tuple(new_data)


Interface.register(GridInterface)
