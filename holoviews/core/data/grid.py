from collections import OrderedDict, defaultdict

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np

from .dictionary import DictInterface
from .interface import Interface
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
    to one another with each key dimension specifiying an axis of the
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

        dimensions = [d.key if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if isinstance(data, tuple):
            data = {d: v for d, v in zip(dimensions, data)}
        elif not isinstance(data, dict):
            raise TypeError('GridInterface must be instantiated as a '
                            'dictionary or tuple')

        for dim in kdims+vdims:
            name = dim.key if isinstance(dim, Dimension) else dim
            if name not in data:
                raise ValueError("Values for dimension %s not found" % dim)
            if not isinstance(data[name], np.ndarray):
                data[name] = np.array(data[name])

        kdim_names = [d.key if isinstance(d, Dimension) else d for d in kdims]
        vdim_names = [d.key if isinstance(d, Dimension) else d for d in vdims]
        expected = tuple([len(data[kd]) for kd in kdim_names])
        for vdim in vdim_names:
            shape = data[vdim].shape
            if shape != expected[::-1] and not (not expected and shape == (1,)):
                raise ValueError('Key dimension values and value array %s '
                                 'shape do not match. Expected shape %s, '
                                 'actual shape: %s' % (vdim, expected[::-1], shape))
        return data, {'kdims':kdims, 'vdims':vdims}, {}


    @classmethod
    def validate(cls, dataset):
        Interface.validate(dataset)


    @classmethod
    def dimension_type(cls, dataset, dim):
        if dim in dataset.dimensions():
            arr = cls.values(dataset, dim, False, False)
        else:
            return None
        return arr.dtype.type


    @classmethod
    def shape(cls, dataset):
        return cls.length(dataset), len(dataset.dimensions()),


    @classmethod
    def length(cls, dataset):
        return np.product([len(dataset.data[d.key]) for d in dataset.kdims])


    @classmethod
    def coords(cls, dataset, dim, ordered=False, expanded=False):
        """
        Returns the coordinates along a dimension.  Ordered ensures
        coordinates are in ascending order and expanded creates
        ND-array matching the dimensionality of the dataset.
        """
        dim = dataset.get_dimension(dim)
        if expanded:
            return util.expand_grid_coords(dataset, dim)
        data = dataset.data[dim.key]
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
            coord_dims = dataset.dimensions('key', label='key')[::-1]

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
        data = data.__getitem__(slices) if invert else data

        # Transpose data
        dims = [name for name in coord_dims[::-1]
                if isinstance(cls.coords(dataset, name), np.ndarray)]
        dropped = [dims.index(d) for d in dims if d not in dataset.kdims]
        inds = [dims.index(kd.key) for kd in dataset.kdims]
        inds += dropped
        if inds:
            data = data.transpose(inds)

        # Allow lower dimensional views into data
        if len(dataset.kdims) < 2:
            data = data.flatten()
        elif dropped:
            data = data.squeeze(axis=tuple(range(len(dropped))))
        return data


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True):
        dim = dataset.get_dimension(dim)
        if dim in dataset.vdims:
            data = dataset.data.get(dim.key)
            data = cls.canonicalize(dataset, data)
            return data.T.flatten() if flat else data
        elif expanded:
            data = cls.coords(dataset, dim.key, expanded=True)
            return data.flatten() if flat else data
        else:
            return cls.coords(dataset, dim.key, ordered=True)


    @classmethod
    def groupby(cls, dataset, dim_names, container_type, group_type, **kwargs):
        # Get dimensions information
        dimensions = [dataset.get_dimension(d) for d in dim_names]
        kdims = [kdim for kdim in dataset.kdims if kdim not in dimensions]

        # Update the kwargs appropriately for Element group types
        group_kwargs = {}
        group_type = dict if group_type == 'raw' else group_type
        if issubclass(group_type, Element):
            group_kwargs.update(util.get_param_values(dataset))
            group_kwargs['kdims'] = kdims
        group_kwargs.update(kwargs)

        # Find all the keys along supplied dimensions
        keys = [dataset.data[d.key] for d in dimensions]

        # Iterate over the unique entries applying selection masks
        grouped_data = []
        for unique_key in zip(*util.cartesian_product(keys)):
            group_data = cls.select(dataset, **dict(zip(dim_names, unique_key)))
            if np.isscalar(group_data):
                group_data = {dataset.vdims[0].key: np.atleast_1d(group_data)}
                for dim, v in zip(dim_names, unique_key):
                    group_data[dim] = np.atleast_1d(v)
            else:
                for vdim in dataset.vdims:
                    group_data[vdim.key] = np.squeeze(group_data[vdim.key])
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
        selection = [(d, selection.get(d.name, selection.get(d.key)))
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
            data[dim.key] = values
        int_inds = [np.argwhere(v) for v in value_select][::-1]
        index = np.ix_(*[np.atleast_1d(np.squeeze(ind)) if ind.ndim > 1 else np.atleast_1d(ind)
                         for ind in int_inds])
        for vdim in dataset.vdims:
            data[vdim.key] = dataset.data[vdim.key][index]

        if indexed and len(data[dataset.vdims[0].key]) == 1:
            return data[dataset.vdims[0].key][0]

        return data


    @classmethod
    def sample(cls, dataset, samples=[]):
        """
        Samples the gridded data into dataset of samples.
        """
        ndims = dataset.ndims
        dimensions = dataset.dimensions(label='key')
        arrays = [dataset.data[vdim.key] for vdim in dataset.vdims]
        data = defaultdict(list)

        first_sample = util.wrap_tuple(samples[0])
        if any(len(util.wrap_tuple(s)) != len(first_sample) for s in samples):
            raise IndexError('Sample coordinates must all be of the same length.')

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
                data[vdim.key].append(array.flat[flat_index])
        concatenated = {d: np.concatenate(arrays).flatten() for d, arrays in data.items()}
        return concatenated


    @classmethod
    def aggregate(cls, dataset, kdims, function, **kwargs):
        kdims = [kd.key if isinstance(kd, Dimension) else kd for kd in kdims]
        data = {kdim: dataset.data[kdim] for kdim in kdims}
        axes = tuple(dataset.ndims-dataset.get_dimension_index(kdim)-1
                     for kdim in dataset.kdims if kdim not in kdims)
        for vdim in dataset.vdims:
            data[vdim.key] = np.atleast_1d(function(dataset.data[vdim.key],
                                                      axis=axes, **kwargs))

        return data


    @classmethod
    def reindex(cls, dataset, kdims, vdims):
        dropped_kdims = [kd for kd in dataset.kdims if kd not in kdims]
        if dropped_kdims and any(len(dataset.data[kd.key]) > 1 for kd in dropped_kdims):
            raise ValueError('Compressed format does not allow dropping key dimensions '
                             'which are not constant.')
        if (any(kd for kd in kdims if kd not in dataset.kdims) or
            any(vd for vd in vdims if vd not in dataset.vdims)):
            return dataset.clone(dataset.columns()).reindex(kdims, vdims)
        dropped_vdims = ([vdim for vdim in dataset.vdims
                          if vdim not in vdims] if vdims else [])
        data = {k: values for k, values in dataset.data.items()
                if k not in dropped_kdims+dropped_vdims}

        if kdims != dataset.kdims:
            joined_dims = kdims+dropped_kdims
            axes = tuple(dataset.ndims-dataset.kdims.index(d)-1
                         for d in joined_dims)
            dropped_axes = tuple(dataset.ndims-joined_dims.index(d)-1
                                 for d in dropped_kdims)
            for vdim in vdims:
                vdata = data[vdim.key]
                if len(axes) > 1:
                    vdata = vdata.transpose(axes[::-1])
                if dropped_axes:
                    vdata = vdata.squeeze(axis=dropped_axes)
                data[vdim.key] = vdata
        return data


    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        if not vdim:
            raise Exception("Cannot add key dimension to a dense representation.")
        dim = dimension.key if isinstance(dimension, Dimension) else dimension
        return dict(dataset.data, **{dim: values})


    @classmethod
    def sort(cls, dataset, by=[]):
        if not by or by in [dataset.kdims, dataset.dimensions()]:
            return dataset.data
        else:
            raise Exception('Compressed format cannot be sorted, either instantiate '
                            'in the desired order or use the expanded format.')


Interface.register(GridInterface)
