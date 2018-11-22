from __future__ import absolute_import

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np
import param

from .. import util
from ..dimension import redim, Dimension, process_dimensions
from ..element import Element
from ..ndmapping import OrderedDict
from ..spaces import HoloMap, DynamicMap
from .interface import Interface, iloc, ndloc
from .array import ArrayInterface
from .dictionary import DictInterface
from .grid import GridInterface
from .multipath import MultiInterface         # noqa (API import)
from .image import ImageInterface             # noqa (API import)

default_datatype = 'dictionary'
datatypes = ['dictionary', 'grid']

try:
    import pandas as pd # noqa (Availability import)
    from .pandas import PandasInterface
    default_datatype = 'dataframe'
    datatypes = ['dataframe', 'dictionary', 'grid']
    DFColumns = PandasInterface
except ImportError:
    pd = None
except Exception as e:
    pd = None
    param.main.warning('Pandas interface failed to import with '
                       'following error: %s' % e)

try:
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
if 'multitabular' not in datatypes:
    datatypes.append('multitabular')


def concat(datasets, datatype=None):
    """
    Concatenates multiple datasets wrapped in an NdMapping type along
    all of its dimensions. Before concatenation all datasets are cast
    to the same datatype, which may be explicitly defined or
    implicitly derived from the first datatype that is
    encountered. For columnar data concatenation adds the columns for
    the dimensions being concatenated along and then concatenates all
    the old and new columns. For gridded data a new axis is created
    for each dimension being concatenated along and then
    hierarchically concatenates along each dimension.

    Arguments
    ---------
    datasets: NdMapping
       NdMapping of Datasets defining dimensions to concatenate on
    datatype: str
        Datatype to cast data to before concatenation, e.g. 'dictionary',
        'dataframe', etc.

    Returns
    -------
    dataset: Dataset
        Concatenated dataset
    """
    return Interface.concatenate(datasets, datatype)


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
            if pd and issubclass(self._element.interface, PandasInterface):
                ds_dims = self._element.dimensions()
                ds_kdims = [self._element.get_dimension(d) if d in ds_dims else d
                            for d in groupby+kdims]
                ds_vdims = [self._element.get_dimension(d) if d in ds_dims else d
                            for d in vdims]
                selected = self._element.clone(kdims=ds_kdims, vdims=ds_vdims)
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

    The Dataset class supports various methods offering a consistent
    way of working with the stored data regardless of the storage
    format used. These operations include indexing, selection and
    various ways of aggregating or collapsing the data with a supplied
    function.
    """

    datatype = param.List(datatypes, doc="""
        A priority list of the data types to be used for storage
        on the .data attribute. If the input supplied to the element
        constructor cannot be put into the requested format, the next
        format listed will be used until a suitable format is found (or
        the data fails to be understood).""")

    group = param.String(default='Dataset', constant=True)

    # In the 1D case the interfaces should not automatically add x-values
    # to supplied data
    _auto_indexable_1d = False

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

        validate_vdims = kwargs.pop('_validate_vdims', True)
        initialized = Interface.initialize(type(self), data, kdims, vdims,
                                           datatype=kwargs.get('datatype'))
        (data, self.interface, dims, extra_kws) = initialized
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

        Arguments
        ---------
        coords: list (optional)
            List of nd-coordinates
        **kwargs: dictionary
            Coordinates specified as keyword pairs of dimension and
            coordinate

        Returns
        -------
        closest: list
            List of tuples of the snapped coordinates

        Raises
        ------
        NotImplementedError:
            Raised if the element does not implement snapping
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


    def sort(self, by=None, reverse=False):
        """
        Sorts the data by the values along the supplied dimensions.

        Arguments
        ---------
        by: list
            List of dimensions to sort by
        reverse: boolean (optional, default=False)
            Whether to sort in reverse order

        Returns
        -------
        dataset: Dataset
            Element of the same type sorted along the specified dimensions
        """
        if by is None:
            by = self.kdims
        elif not isinstance(by, list):
            by = [by]
        sorted_columns = self.interface.sort(self, by, reverse)
        return self.clone(sorted_columns)


    def range(self, dim, data_range=True, dimension_range=True):
        """
        Returns the range of values along the specified dimension.

        Arguments
        ---------
        dimension: Dimension, str or int
            The dimension to compute the range on.
        data_range: bool (optional, default=True)
            Whether the range should include the data range or only
            the dimension ranges
        dimension_range: bool (optional, True)
            Whether to compute the range including the Dimension range
            and soft_range

        Returns
        -------
        range: tuple
            Tuple of length two containing the lower and upper bound
        """
        dim = self.get_dimension(dim)

        if dim is None or (not data_range and not dimension_range):
            return (None, None)
        elif all(util.isfinite(v) for v in dim.range) and dimension_range:
            return dim.range
        elif dim in self.dimensions() and data_range and len(self):
            lower, upper = self.interface.range(self, dim)
        else:
            lower, upper = (np.NaN, np.NaN)
        if not dimension_range:
            return lower, upper
        return util.dimension_range(lower, upper, dim.range, dim.soft_range)


    def add_dimension(self, dimension, dim_pos, dim_val, vdim=False, **kwargs):
        """
        Create a new object with an additional key dimensions.  Requires
        the dimension name or object, the desired position in the key
        dimensions and a key value scalar or sequence of the same length
        as the existing keys.

        Arguments
        ---------
        dimension: Dimension, str or tuple
            Dimension or dimension spec to add
        dim_pos: int
            Integer index to insert dimension at
        dim_val: scalar or numpy.ndarray
            Dimension value(s) to add
        vdim: bool (optional, default=False)
            Whether the dimension is a value dimension
        **kwargs:
            Keyword arguments passed to the cloned element

        Returns
        -------
        clone: Dataset
            Cloned Dataset containing the new dimension
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
            element = self.clone(datatype=[default_datatype])
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

        Arguments
        ---------
        selection_specs: list
            A list of types, functions, or type[.group][.label] strings
            specifying which objects to apply the selection on
        **selection: dict
            Selection to apply mapping from dimension name to selection.
            Selections can be scalar values, tuple ranges, lists of
            discrete values and boolean arrays

        Returns
        -------
        selection: Dataset or scalar
            Returns an element containing the selected data or a scalar
            if a single value was selected
        """
        if not isinstance(selection_specs, (list, tuple, set)):
            selection_specs = [selection_specs]
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

        Arguments
        ---------
        kdims: list
            List of key dimensions
        vdims: list
            List of value dimensions

        Returns
        -------
        reindexed: Dataset
            Reindexed Dataset
        """

        gridded = self.interface.gridded
        scalars = []
        if gridded:
            coords = [(d, self.interface.coords(self, d.name)) for d in self.kdims]
            scalars = [d for d, vs in coords if len(vs) == 1]

        if kdims is None:
            # If no key dimensions are defined and interface is gridded
            # drop all scalar key dimensions
            key_dims = [d for d in self.kdims if (not vdims or d not in vdims)
                        and not d in scalars]
        elif not isinstance(kdims, list):
            key_dims = [self.get_dimension(kdims, strict=True)]
        else:
            key_dims = [self.get_dimension(k, strict=True) for k in kdims]
        dropped = [d for d in self.kdims if not d in key_dims and not d in scalars]

        new_type = None
        if vdims is None:
            val_dims = [d for d in self.vdims if not kdims or d not in kdims]
        else:
            val_dims = [self.get_dimension(v, strict=True) for v in vdims]
            new_type = self._vdim_reductions.get(len(val_dims), type(self))

        data = self.interface.reindex(self, key_dims, val_dims)
        datatype = self.datatype
        if gridded and dropped:
            datatype = [dt for dt in datatype if not self.interface.interfaces[dt].gridded]
        return self.clone(data, kdims=key_dims, vdims=val_dims,
                          new_type=new_type, datatype=datatype)


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
        just the selected samples. Supports two signatures:

        Sampling with a list of coordinates, e.g.:

            ds.sample([(0, 0), (0.1, 0.2), ...])

        Sampling by keyword, e.g.:

            ds.sample(x=0)

        Arguments
        ---------
        samples: list (optional)
            List of nd-coordinates to sample
        closest: bool (optional, default=True)
            Whether to snap to the closest coordinate (if the Element supports it)
        **kwargs: dict (optional)
            Keywords of dimensions and scalar coordinates

        Returns
        -------
        sampled: Curve or Table
            Element containing the sampled coordinates
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
                reindexed = selection.clone(new_type=Dataset).reindex(kdims)
                selection = tuple(reindexed.columns(kdims+self.vdims).values())

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


    def reduce(self, dimensions=[], function=None, spreadfn=None, **reductions):
        """
        Allows reducing the values along one or more key dimension with
        the supplied function (reciprocal operation to aggregate).
        Supports two signatures:

        Reducing with a list of coordinates, e.g.:

            ds.reduce(['x'], np.mean)

        Sampling by keyword, e.g.:

            ds.reduce(x=np.mean)

        Arguments
        ---------
        dimensions: Dimension/str or list (optional)
            Dimension or list of dimensions to aggregate on, defaults
            to all current key dimensions
        function: function (optional)
            Function to compute aggregate with, e.g. numpy.mean
        spreadfn: function (optional)
            Function to compute a secondary aggregate, e.g. to compute
            a confidence interval, spread, or standard deviation
        **reductions:
            Reductions specified as keyword pairs of the dimension name
            and reduction function, e.g. Dataset.reduce(x=np.mean)

        Returns
        -------
        reduced: Dataset
            Returns the reduced Dataset
        """
        if any(dim in self.vdims for dim in dimensions):
            raise Exception("Reduce cannot be applied to value dimensions")
        function, dims = self._reduce_map(dimensions, function, reductions)
        dims = [d for d in self.kdims if d not in dims]
        return self.aggregate(dims, function, spreadfn)


    def aggregate(self, dimensions=None, function=None, spreadfn=None, **kwargs):
        """
        Aggregates over the supplied key dimensions with the defined
        function.

        Arguments
        ---------
        dimensions: Dimension/str or list (optional)
            Dimension or list of dimensions to aggregate on, defaults
            to all current key dimensions
        function: function
            Function to compute aggregate with, e.g. numpy.mean
        spreadfn: function (optional)
            Function to compute a secondary aggregate, e.g. to compute
            a confidence interval, spread, or standard deviation
        **kwargs:
            Keyword arguments passed to the aggregation function

        Returns
        -------
        aggregated: Dataset
            Returns the aggregated Dataset
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
        """
        Return the results of a groupby operation over the specified
        dimensions as an object of type container_type (expected to be
        dictionary-like).

        Arguments
        ---------
        dimensions: Dimension/str or list
            Dimension or list of dimensions to group by
        container_type: NdMapping, list or dict (optional, default=HoloMap)
            Container type to wrap groups in
        group_type: Element type (optional)
            If supplied casts each group to this type
        dynamic: bool (optional, default=False)
            Whether to apply dynamic groupby and return a DynamicMap
        **kwargs:
            Keyword arguments to pass to each group

        Returns
        -------
        grouped: container_type
            Returns object of supplied container_type containing the
            groups. If dynamic=True returns a DynamicMap instead.
        """
        if not isinstance(dimensions, list): dimensions = [dimensions]
        if not len(dimensions): dimensions = self.dimensions('key', True)
        if group_type is None: group_type = type(self)

        dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        dim_names = [d.name for d in dimensions]

        if dynamic:
            group_dims = [kd for kd in self.kdims if kd not in dimensions]
            kdims = [self.get_dimension(d) for d in kwargs.pop('kdims', group_dims)]
            drop_dim = len(group_dims) != len(kdims)
            group_kwargs = dict(util.get_param_values(self), kdims=kdims)
            group_kwargs.update(kwargs)
            def load_subset(*args):
                constraint = dict(zip(dim_names, args))
                group = self.select(**constraint)
                if np.isscalar(group):
                    return group_type(([group],), group=self.group,
                                      label=self.label, vdims=self.vdims)
                data = group.reindex(kdims)
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


    def dimension_values(self, dimension, expanded=True, flat=True):
        """
        Returns the values along a particular dimension. If unique
        values are requested will return only unique values.

        Arguments
        ---------
        dimension: Dimension, str or int
            The dimension to query values on
        expanded: boolean (optional, default=True)
            Whether to return the expanded values, behavior depends
            on the type of data:
              - Columnar: If false returns unique values
              - Geometry: If false returns scalar values per geometry
              - Gridded: If false returns 1D coordinates
        flat: boolean (optional, default=True)
            Whether the array should be flattened to a 1D array

        Returns
        -------
        array: numpy.ndarray
            NumPy array of values along the requested dimension
        """
        dim = self.get_dimension(dim, strict=True)
        return self.interface.values(self, dim, expanded, flat)


    def get_dimension_type(self, dim):
        """
        Returns the specified Dimension type if specified or
        if the dimension_values types are consistent.

        Arguments
        ---------
        dimension: Dimension, str or int
            Dimension to look up by name or by index

        Returns
        -------
        dimension_type: int
            Declared type of values along the dimension
        """
        dim_obj = self.get_dimension(dim)
        if dim_obj and dim_obj.type is not None:
            return dim_obj.type
        return self.interface.dimension_type(self, dim_obj)


    def dframe(self, dimensions=None, multi_index=False):
        """
        Returns a pandas dataframe of columns along each dimension.

        Arguments
        ---------
        dimensions: list (optional)
            List of dimensions to return (defaults to all dimensions)
        multi_index: boolean (optional, default=False)
            Whether to treat key dimensions as (multi-)indexes

        Returns
        -------
        dataframe: pandas.DataFrame
            DataFrame of columns corresponding to each dimension
        """
        if dimensions is None:
            dimensions = [d.name for d in self.dimensions()]
        else:
            dimensions = [self.get_dimension(d, strict=True).name for d in dimensions]
        df = self.interface.dframe(self, dimensions)
        if multi_index:
            df = df.set_index([d for d in dimensions if d in self.kdims])
        return df


    def columns(self, dimensions=None):
        """
        Returns a dictionary of column arrays along each dimension
        of the element.

        Arguments
        ---------
        dimensions: list (optional)
            List of dimensions to return (defaults to all dimensions)

        Returns
        -------
        columns: OrderedDict
            Dictionary of arrays for each dimension
        """
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


    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        """
        Returns a clone of the object with matching parameter values
        containing the specified args and kwargs.

        If shared_data is set to True and no data explicitly supplied,
        the clone will share data with the original. May also supply
        a new_type, which will inherit all shared parameters.

        Arguments
        ---------
        data: valid data format, e.g. a tuple of arrays (optional)
            The data to replace existing data with
        shared_data: bool (optional, default=True)
            Whether to use the existing data
        new_type: Element type
            An Element type to cast the clone to
        *args:
            Additional arguments
        **overrides:
            Additional keyword arguments to pass to cloned constructor

        Returns
        -------
        clone: Dataset instance
            Cloned Dataset instance
        """
        if 'datatype' not in overrides:
            datatypes = [self.interface.datatype] + self.datatype
            overrides['datatype'] = list(util.unique_iterator(datatypes))
        return super(Dataset, self).clone(data, shared_data, new_type, *args, **overrides)


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
