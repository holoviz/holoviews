from __future__ import absolute_import

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np
import param

from ..dimension import redim
from ..util import dimension_range
from .interface import Interface, iloc, ndloc
from .array import ArrayInterface
from .dictionary import DictInterface
from .grid import GridInterface
from .multipath import MultiInterface         # noqa (API import)
from .image import ImageInterface             # noqa (API import)

datatypes = ['dictionary', 'grid']

try:
    import pandas as pd # noqa (Availability import)
    from .pandas import PandasInterface
    datatypes = ['dataframe', 'dictionary', 'grid', 'ndelement', 'array']
    DFColumns = PandasInterface
except ImportError:
    pass
except Exception as e:
    param.main.warning('Pandas interface failed to import with '
                       'following error: %s' % e)

try:
    import iris # noqa (Availability import)
    from .iris import CubeInterface # noqa (Conditional API import)
    datatypes.append('cube')
except ImportError:
    pass
except Exception as e:
    param.main.warning('Iris interface failed to import with '
                       'following error: %s' % e)

try:
    import xarray # noqa (Availability import)
    from .xarray import XArrayInterface # noqa (Conditional API import)
    datatypes.append('xarray')
except ImportError:
    pass

try:
    from .dask import DaskInterface   # noqa (Conditional API import)
    datatypes.append('dask')
except ImportError:
    pass

if 'array' not in datatypes:
    datatypes.append('array')

from ..dimension import Dimension, process_dimensions
from ..element import Element
from ..ndmapping import OrderedDict
from ..spaces import HoloMap, DynamicMap
from .. import util


class DataConversion(object):
    """
    DataConversion is a very simple container object which can be
    given an existing Dataset Element and provides methods to convert
    the Dataset into most other Element types.
    """

    def __init__(self, element):
        self._element = element

    def __call__(self, new_type, kdims=None, vdims=None, groupby=None,
                 sort=False, **kwargs):
        """
        Generic conversion method for Dataset based Element
        types. Supply the Dataset Element type to convert to and
        optionally the key dimensions (kdims), value dimensions
        (vdims) and the dimensions.  to group over. Converted Columns
        can be automatically sorted via the sort option and kwargs can
        be passed through.
        """
        if 'mdims' in kwargs:
            if groupby:
                raise ValueError('Cannot supply both mdims and groupby')
            else:
                self._element.warning("'mdims' keyword has been renamed "
                                      "to 'groupby'; the name mdims is "
                                      "deprecated and will be removed "
                                      "after version 1.7.")
                groupby = kwargs.pop('mdims')

        element_params = new_type.params()
        kdim_param = element_params['kdims']
        vdim_param = element_params['vdims']
        if isinstance(kdim_param.bounds[1], int):
            ndim = min([kdim_param.bounds[1], len(kdim_param.default)])
        else:
            ndim = None
        nvdim = vdim_param.bounds[1] if isinstance(vdim_param.bounds[1], int) else None
        if kdims is None:
            kd_filter = groupby or []
            if not isinstance(kd_filter, list):
                kd_filter = [groupby]
            kdims = [kd for kd in self._element.kdims if kd not in kd_filter][:ndim]
        elif kdims and not isinstance(kdims, list): kdims = [kdims]
        if vdims is None:
            vdims = [d for d in self._element.vdims if d not in kdims][:nvdim]
        if vdims and not isinstance(vdims, list): vdims = [vdims]

        # Checks Element type supports dimensionality
        type_name = new_type.__name__
        for dim_type, dims in (('kdims', kdims), ('vdims', vdims)):
            min_d, max_d = new_type.params(dim_type).bounds
            if ((min_d is not None and len(dims) < min_d) or
                (max_d is not None and len(dims) > max_d)):
                raise ValueError("%s %s must be between length %s and %s." %
                                 (type_name, dim_type, min_d, max_d))

        if groupby is None:
            groupby = [d for d in self._element.kdims if d not in kdims+vdims]
        elif groupby and not isinstance(groupby, list):
            groupby = [groupby]

        if self._element.interface.gridded:
            dropped_kdims = [kd for kd in self._element.kdims if kd not in groupby+kdims]
            if dropped_kdims:
                selected = self._element.reindex(groupby+kdims, vdims)
            else:
                selected = self._element
        else:
            selected = self._element.reindex(groupby+kdims, vdims)
        params = {'kdims': [selected.get_dimension(kd, strict=True) for kd in kdims],
                  'vdims': [selected.get_dimension(vd, strict=True) for vd in vdims],
                  'label': selected.label}
        if selected.group != selected.params()['group'].default:
            params['group'] = selected.group
        params.update(kwargs)
        if len(kdims) == selected.ndims or not groupby:
            element = new_type(selected, **params)
            return element.sort() if sort else element
        group = selected.groupby(groupby, container_type=HoloMap,
                                 group_type=new_type, **params)
        if sort:
            return group.map(lambda x: x.sort(), [new_type])
        else:
            return group



class Dataset(Element):
    """
    Dataset provides a general baseclass for Element types that
    contain structured data and supports a range of data formats.

    The Dataset class supports various methods offering a consistent way
    of working with the stored data regardless of the storage format
    used. These operations include indexing, selection and various ways
    of aggregating or collapsing the data with a supplied function.
    """

    datatype = param.List(datatypes,
        doc=""" A priority list of the data types to be used for storage
        on the .data attribute. If the input supplied to the element
        constructor cannot be put into the requested format, the next
        format listed will be used until a suitable format is found (or
        the data fails to be understood).""")

    group = param.String(default='Dataset', constant=True)

    # In the 1D case the interfaces should not automatically add x-values
    # to supplied data
    _auto_indexable_1d = True

    # Define a class used to transform Datasets into other Element types
    _conversion_interface = DataConversion

    # Whether the key dimensions are specified as bins
    _binned = False

    _vdim_reductions = {}
    _kdim_reductions = {}

    def __init__(self, data, kdims=None, vdims=None, **kwargs):
        if isinstance(data, Element):
            pvals = util.get_param_values(data)
            kwargs.update([(l, pvals[l]) for l in ['group', 'label']
                           if l in pvals and l not in kwargs])
        kwargs.update(process_dimensions(kdims, vdims))
        kdims, vdims = kwargs.get('kdims'), kwargs.get('vdims')

        initialized = Interface.initialize(type(self), data, kdims, vdims,
                                           datatype=kwargs.get('datatype'))
        (data, self.interface, dims, extra_kws) = initialized
        validate_vdims = kwargs.pop('_validate_vdims', True)
        super(Dataset, self).__init__(data, **dict(kwargs, **dict(dims, **extra_kws)))
        self.interface.validate(self, validate_vdims)

        self.redim = redim(self, mode='dataset')


    def closest(self, coords=[], **kwargs):
        """
        Given a single coordinate or multiple coordinates as
        a tuple or list of tuples or keyword arguments matching
        the dimension closest will find the closest actual x/y
        coordinates. Different Element types should implement this
        appropriately depending on the space they represent, if the
        Element does not support snapping raise NotImplementedError.
        """
        if self.ndims > 1:
            raise NotImplementedError("Closest method currently only "
                                      "implemented for 1D Elements")

        if kwargs:
            if len(kwargs) > 1:
                raise NotImplementedError("Closest method currently only "
                                          "supports 1D indexes")
            samples = list(kwargs.values())[0]
            coords = samples if isinstance(samples, list) else [samples]

        xs = self.dimension_values(0)
        if xs.dtype.kind in 'SO':
            raise NotImplementedError("Closest only supported for numeric types")
        idxs = [np.argmin(np.abs(xs-coord)) for coord in coords]
        return [xs[idx] for idx in idxs]


    def sort(self, by=[], reverse=False):
        """
        Sorts the data by the values along the supplied dimensions.
        """
        if not by: by = self.kdims
        if not isinstance(by, list): by = [by]

        sorted_columns = self.interface.sort(self, by, reverse)
        return self.clone(sorted_columns)


    def range(self, dim, data_range=True):
        """
        Computes the range of values along a supplied dimension, taking
        into account the range and soft_range defined on the Dimension
        object.
        """
        dim = self.get_dimension(dim)
        if dim is None:
            return (None, None)
        elif all(v is not None and np.isfinite(v) for v in dim.range):
            return dim.range
        elif dim in self.dimensions() and data_range and len(self):
            lower, upper = self.interface.range(self, dim)
        else:
            lower, upper = (np.NaN, np.NaN)
        return dimension_range(lower, upper, dim)


    def add_dimension(self, dimension, dim_pos, dim_val, vdim=False, **kwargs):
        """
        Create a new object with an additional key dimensions.  Requires
        the dimension name or object, the desired position in the key
        dimensions and a key value scalar or sequence of the same length
        as the existing keys.
        """
        if isinstance(dimension, (util.basestring, tuple)):
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

        if issubclass(self.interface, ArrayInterface) and np.asarray(dim_val).dtype != self.data.dtype:
            element = self.clone(datatype=['pandas', 'dictionary'])
            data = element.interface.add_dimension(element, dimension, dim_pos, dim_val, vdim)
        else:
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
        selection = {dim: sel for dim, sel in selection.items()
                     if dim in self.dimensions()+['selection_mask']}
        if (selection_specs and not any(self.matches(sp) for sp in selection_specs)
            or not selection):
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
            key_dims = [d for d in self.kdims if not vdims or d not in vdims]
        else:
            key_dims = [self.get_dimension(k, strict=True) for k in kdims]

        new_type = None
        if vdims is None:
            val_dims = [d for d in self.vdims if not kdims or d not in kdims]
        else:
            val_dims = [self.get_dimension(v, strict=True) for v in vdims]
            new_type = self._vdim_reductions.get(len(val_dims), type(self))

        data = self.interface.reindex(self, key_dims, val_dims)
        return self.clone(data, kdims=key_dims, vdims=val_dims,
                          new_type=new_type)


    def __getitem__(self, slices):
        """
        Allows slicing and selecting values in the Dataset object.
        Supports multiple indexing modes:

           (1) Slicing and indexing along the values of each dimension
               in the columns object using either scalars, slices or
               sets of values.
           (2) Supplying the name of a dimension as the first argument
               will return the values along that dimension as a numpy
               array.
           (3) Slicing of all key dimensions and selecting a single
               value dimension by name.
           (4) A boolean array index matching the length of the Dataset
               object.
        """
        slices = util.process_ellipses(self, slices, vdim_selection=True)
        if isinstance(slices, np.ndarray) and slices.dtype.kind == 'b':
            if not len(slices) == len(self):
                raise IndexError("Boolean index must match length of sliced object")
            return self.clone(self.select(selection_mask=slices))
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
            raise IndexError("%r is not an available value dimension" % slices[self.ndims])
        else:
            selection = dict(zip(self.dimensions(label=True), slices))
        data = self.select(**selection)
        if value_select:
            if data.shape[0] == 1:
                return data[value_select][0]
            else:
                return data.reindex(vdims=[value_select])
        return data


    def sample(self, samples=[], closest=True, **kwargs):
        """
        Allows sampling of Dataset as an iterator of coordinates
        matching the key dimensions, returning a new object containing
        just the selected samples. Alternatively may supply kwargs
        to sample a coordinate on an object. By default it will attempt
        to snap to the nearest coordinate if the Element supports it,
        snapping may be disabled with the closest argument.
        """
        if kwargs and samples:
            raise Exception('Supply explicit list of samples or kwargs, not both.')
        elif kwargs:
            sample = [slice(None) for _ in range(self.ndims)]
            for dim, val in kwargs.items():
                sample[self.get_dimension_index(dim)] = val
            samples = [tuple(sample)]

        # Note: Special handling sampling of gridded 2D data as Curve
        # may be replaced with more general handling
        # see https://github.com/ioam/holoviews/issues/1173
        from ...element import Table, Curve
        if len(samples) == 1:
            sel = {kd.name: s for kd, s in zip(self.kdims, samples[0])}
            dims = [kd for kd, v in sel.items() if not np.isscalar(v)]
            selection = self.select(**sel)

            # If a 1D cross-section of 2D space return Curve
            if self.interface.gridded and self.ndims == 2 and len(dims) == 1:
                new_type = Curve
                kdims = [self.get_dimension(kd) for kd in dims]
            else:
                new_type = Table
                kdims = self.kdims

            if np.isscalar(selection):
                selection = [samples[0]+(selection,)]
            else:
                selection = tuple(selection.columns(kdims+self.vdims).values())

            datatype = list(util.unique_iterator(self.datatype+['dataframe', 'dict']))
            return self.clone(selection, kdims=kdims, new_type=new_type,
                              datatype=datatype)

        lens = set(len(util.wrap_tuple(s)) for s in samples)
        if len(lens) > 1:
            raise IndexError('Sample coordinates must all be of the same length.')

        if closest:
            try:
                samples = self.closest(samples)
            except NotImplementedError:
                pass
        samples = [util.wrap_tuple(s) for s in samples]
        return self.clone(self.interface.sample(self, samples), new_type=Table)


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


    def aggregate(self, dimensions=None, function=None, spreadfn=None, **kwargs):
        """
        Aggregates over the supplied key dimensions with the defined
        function.
        """
        if function is None:
            raise ValueError("The aggregate method requires a function to be specified")
        if dimensions is None: dimensions = self.kdims
        elif not isinstance(dimensions, list): dimensions = [dimensions]
        kdims = [self.get_dimension(d, strict=True) for d in dimensions]
        if not len(self):
            if spreadfn:
                spread_name = spreadfn.__name__
                vdims = [d for vd in self.vdims for d in [vd, vd('_'.join([vd.name, spread_name]))]]
            else:
                vdims = self.vdims
            return self.clone([], kdims=kdims, vdims=vdims)

        aggregated = self.interface.aggregate(self, kdims, function, **kwargs)
        aggregated = self.interface.unpack_scalar(self, aggregated)

        ndims = len(dimensions)
        min_d, max_d = self.params('kdims').bounds
        generic_type = (min_d is not None and ndims < min_d) or (max_d is not None and ndims > max_d)

        vdims = self.vdims
        if spreadfn:
            error = self.interface.aggregate(self, dimensions, spreadfn)
            spread_name = spreadfn.__name__
            ndims = len(vdims)
            error = self.clone(error, kdims=kdims, new_type=Dataset)
            combined = self.clone(aggregated, kdims=kdims, new_type=Dataset)
            for i, d in enumerate(vdims):
                dim = d('_'.join([d.name, spread_name]))
                dvals = error.dimension_values(d, flat=False)
                combined = combined.add_dimension(dim, ndims+i, dvals, True)
            return combined.clone(new_type=Dataset if generic_type else type(self))

        if np.isscalar(aggregated):
            return aggregated
        else:
            try:
                # Should be checking the dimensions declared on the element are compatible
                return self.clone(aggregated, kdims=kdims, vdims=vdims)
            except:
                datatype = self.params('datatype').default
                return self.clone(aggregated, kdims=kdims, vdims=vdims,
                                  new_type=Dataset if generic_type else None,
                                  datatype=datatype)


    def groupby(self, dimensions=[], container_type=HoloMap, group_type=None,
                dynamic=False, **kwargs):
        """Return the results of a groupby operation over the specified
        dimensions as an object of type container_type (expected to be
        dictionary-like).

        Keys vary over the columns (dimensions) and the corresponding
        values are collections of group_type (e.g an Element, list, tuple)
        constructed with kwargs (if supplied).

        If dynamic is requested container_type is automatically set to
        a DynamicMap, allowing dynamic exploration of large
        datasets. If the data does not represent a full cartesian grid
        of the requested dimensions some Elements will be empty.
        """
        if not isinstance(dimensions, list): dimensions = [dimensions]
        if not len(dimensions): dimensions = self.dimensions('key', True)
        if group_type is None: group_type = type(self)

        dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        dim_names = [d.name for d in dimensions]

        if dynamic:
            group_dims = [d.name for d in self.kdims if d not in dimensions]
            kdims = [self.get_dimension(d) for d in group_dims]
            group_kwargs = dict(util.get_param_values(self), kdims=kdims)
            group_kwargs.update(kwargs)
            drop_dim = len(kdims) != len(group_kwargs['kdims'])
            def load_subset(*args):
                constraint = dict(zip(dim_names, args))
                group = self.select(**constraint)
                if np.isscalar(group):
                    return group_type(([group],), group=self.group,
                                      label=self.label, vdims=self.vdims)
                data = group.reindex(group_dims)
                if drop_dim and self.interface.gridded:
                    data = data.columns()
                return group_type(data, **group_kwargs)
            dynamic_dims = [d(values=list(self.interface.values(self, d.name, False)))
                            for d in dimensions]
            return DynamicMap(load_subset, kdims=dynamic_dims)

        return self.interface.groupby(self, dim_names, container_type,
                                      group_type, **kwargs)

    def __len__(self):
        """
        Returns the number of rows in the Dataset object.
        """
        return self.interface.length(self)

    def __nonzero__(self):
        return self.interface.nonzero(self)

    __bool__ = __nonzero__

    @property
    def shape(self):
        "Returns the shape of the data."
        return self.interface.shape(self)


    def dimension_values(self, dim, expanded=True, flat=True):
        """
        Returns the values along a particular dimension. If unique
        values are requested will return only unique values.
        """
        dim = self.get_dimension(dim, strict=True)
        return self.interface.values(self, dim, expanded, flat)


    def get_dimension_type(self, dim):
        """
        Returns the specified Dimension type if specified or
        if the dimension_values types are consistent otherwise
        None is returned.
        """
        dim_obj = self.get_dimension(dim)
        if dim_obj and dim_obj.type is not None:
            return dim_obj.type
        return self.interface.dimension_type(self, dim_obj)


    def dframe(self, dimensions=None):
        """
        Returns the data in the form of a DataFrame. Supplying a list
        of dimensions filters the dataframe. If the data is already
        a DataFrame a copy is returned.
        """
        if dimensions:
            dimensions = [self.get_dimension(d, strict=True).name for d in dimensions]
        return self.interface.dframe(self, dimensions)


    def columns(self, dimensions=None):
        if dimensions is None:
            dimensions = self.dimensions()
        else:
            dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        return OrderedDict([(d.name, self.dimension_values(d)) for d in dimensions])


    @property
    def to(self):
        """
        Property to create a conversion interface with methods to
        convert to other Element types.
        """
        return self._conversion_interface(self)


    @property
    def iloc(self):
        """
        Returns an iloc object providing a convenient interface to
        slice and index into the Dataset using row and column indices.
        Allow selection by integer index, slice and list of integer
        indices and boolean arrays.

        Examples:

        * Index the first row and column:

            dataset.iloc[0, 0]

        * Select rows 1 and 2 with a slice:

            dataset.iloc[1:3, :]

        * Select with a list of integer coordinates:

            dataset.iloc[[0, 2, 3]]
        """
        return iloc(self)


    @property
    def ndloc(self):
        """
        Returns an ndloc object providing nd-array like indexing for
        gridded datasets. Follows NumPy array indexing conventions,
        allowing for indexing, slicing and selecting a list of indices
        on multi-dimensional arrays using integer indices. The order
        of array indices is inverted relative to the Dataset key
        dimensions, e.g. an Image with key dimensions 'x' and 'y' can
        be indexed with ``image.ndloc[iy, ix]``, where ``iy`` and
        ``ix`` are integer indices along the y and x dimensions.

        Examples:

        * Index value in 2D array:

            dataset.ndloc[3, 1]

        * Slice along y-axis of 2D array:

            dataset.ndloc[2:5, :]

        * Vectorized (non-orthogonal) indexing along x- and y-axes:

            dataset.ndloc[[1, 2, 3], [0, 2, 3]]
        """
        return ndloc(self)


# Aliases for pickle backward compatibility
Columns      = Dataset
ArrayColumns = ArrayInterface
DictColumns  = DictInterface
GridColumns  = GridInterface
