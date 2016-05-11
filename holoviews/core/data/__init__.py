from __future__ import absolute_import
from collections import OrderedDict

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np
import param

from .interface import Interface
from .array import ArrayInterface
from .dictionary import DictInterface
from .grid import GridInterface
from .ndelement import NdElementInterface

datatypes = ['array', 'dictionary', 'grid', 'ndelement']

try:
    import pandas as pd # noqa (Availability import)
    from .pandas import PandasInterface
    datatypes = ['array', 'dataframe', 'dictionary', 'grid', 'ndelement']
    DFColumns = PandasInterface
except ImportError:
    pass

try:
    import iris # noqa (Availability import)
    from .iris import CubeInterface # noqa (Conditional API import)
    datatypes.append('cube')
except ImportError:
    pass

from ..dimension import Dimension
from ..element import Element
from ..spaces import HoloMap
from .. import util


class DataConversion(object):
    """
    DataConversion is a very simple container object which can be
    given an existing Dataset Element and provides methods to convert
    the Dataset into most other Element types.
    """

    def __init__(self, element):
        self._element = element

    def __call__(self, new_type, kdims=None, vdims=None, mdims=None,
                 sort=False, **kwargs):
        """
        Generic conversion method for Column based types. Supply the
        Columns based type to convert to and optionally the
        key dimensions (kdims), value dimensions (vdims) and HoloMap
        key dimensions (mdims). Converted Columns can be automatically
        sorted via the sort option and kwargs can be passed through.
        """
        if kdims is None:
            kdims = self._element.kdims
        elif kdims and not isinstance(kdims, list): kdims = [kdims]
        if vdims is None:
            vdims = self._element.vdims
        if vdims and not isinstance(vdims, list): vdims = [vdims]
        if mdims is None:
            mdims = [d for d in self._element.kdims if d not in kdims+vdims]

        selected = self._element.reindex(mdims+kdims, vdims)
        params = {'kdims': [selected.get_dimension(kd) for kd in kdims],
                  'vdims': [selected.get_dimension(vd) for vd in vdims],
                  'label': selected.label}
        if selected.group != selected.params()['group'].default:
            params['group'] = selected.group
        params.update(kwargs)
        if len(kdims) == selected.ndims:
            element = new_type(selected, **params)
            return element.sort() if sort else element
        group = selected.groupby(mdims, container_type=HoloMap,
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

    # In the 1D case the interfaces should not automatically add x-values
    # to supplied data
    _1d = False

    # Define a class used to transform Datasets into other Element types
    _conversion_interface = DataConversion

    def __init__(self, data, **kwargs):
        if isinstance(data, Element):
            pvals = util.get_param_values(data)
            kwargs.update([(l, pvals[l]) for l in ['group', 'label']
                           if l in pvals and l not in kwargs])
        initialized = Interface.initialize(type(self), data,
                                           kwargs.get('kdims'),
                                           kwargs.get('vdims'),
                                           datatype=kwargs.get('datatype'))
        (data, self.interface, dims, extra_kws) = initialized
        super(Dataset, self).__init__(data, **dict(extra_kws, **dict(kwargs, **dims)))
        self.interface.validate(self)


    def __setstate__(self, state):
        """
        Restores OrderedDict based Dataset objects, converting them to
        the up-to-date NdElement format.
        """
        self.__dict__ = state
        if isinstance(self.data, OrderedDict):
            self.data = Dataset(self.data, kdims=self.kdims,
                                vdims=self.vdims, group=self.group,
                                label=self.label)
            self.interface = NdColumns
        elif isinstance(self.data, np.ndarray):
            self.interface = ArrayInterface
        elif util.is_dataframe(self.data):
            self.interface = PandasInterface

        super(Dataset, self).__setstate__(state)

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
        if selection_specs and not any(self.matches(sp) for sp in selection_specs):
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
            key_dims = [self.get_dimension(k) for k in kdims]

        if vdims is None:
            val_dims = [d for d in self.vdims if not kdims or d not in kdims]
        else:
            val_dims = [self.get_dimension(v) for v in vdims]

        data = self.interface.reindex(self, key_dims, val_dims)
        return self.clone(data, kdims=key_dims, vdims=val_dims)


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
        Allows sampling of Dataset as an iterator of coordinates
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


    def aggregate(self, dimensions=None, function=None, spreadfn=None, **kwargs):
        """
        Aggregates over the supplied key dimensions with the defined
        function.
        """
        if function is None:
            raise ValueError("The aggregate method requires a function to be specified")
        if dimensions is None: dimensions = self.kdims
        elif not isinstance(dimensions, list): dimensions = [dimensions]
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

        dimensions = [self.get_dimension(d, strict=True).name for d in dimensions]
        invalid_dims = list(set(dimensions) - set(self.dimensions('key', True)))
        if invalid_dims:
            raise Exception('Following dimensions could not be found:\n%s.'
                            % invalid_dims)
        return self.interface.groupby(self, dimensions, container_type,
                                      group_type, **kwargs)

    def __len__(self):
        """
        Returns the number of rows in the Dataset object.
        """
        return self.interface.length(self)


    @property
    def shape(self):
        "Returns the shape of the data."
        return self.interface.shape(self)


    def dimension_values(self, dim, expanded=True, flat=True):
        """
        Returns the values along a particular dimension. If unique
        values are requested will return only unique values.
        """
        dim = self.get_dimension(dim, strict=True).name
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
        return self.interface.dimension_type(self, dim)


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


    @property
    def to(self):
        """
        Property to create a conversion interface with methods to
        convert to other Element types.
        """
        return self._conversion_interface(self)



# Aliases for pickle backward compatibility
Columns      = Dataset
ArrayColumns = ArrayInterface
DictColumns  = DictInterface
NdColumns    = NdElementInterface
GridColumns  = GridInterface
