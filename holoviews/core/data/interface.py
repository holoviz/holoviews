import sys
import warnings

import param
import numpy as np

from .. import util
from ..element import Element
from ..ndmapping import NdMapping
from .util import finite_range


class DataError(ValueError):
    "DataError is raised when the data cannot be interpreted"

    def __init__(self, msg, interface=None):
        if interface is not None:
            msg = '\n\n'.join([msg, interface.error()])
        super().__init__(msg)


class Accessor:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        from ..data import Dataset
        from ...operation.element import method
        in_method = self.dataset._in_method
        if not in_method:
            self.dataset._in_method = True
        try:
            res = self._perform_getitem(self.dataset, index)
            if not in_method and isinstance(res, Dataset):
                getitem_op = method.instance(
                    input_type=type(self),
                    output_type=type(self.dataset),
                    method_name='_perform_getitem',
                    args=[index],
                )
                res._pipeline = self.dataset.pipeline.instance(
                    operations=self.dataset.pipeline.operations + [getitem_op],
                    output_type=type(self.dataset)
                )
        finally:
            if not in_method:
                self.dataset._in_method = False
        return res

    @classmethod
    def _perform_getitem(cls, dataset, index):
        raise NotImplementedError()


class iloc(Accessor):
    """
    iloc is small wrapper object that allows row, column based
    indexing into a Dataset using the ``.iloc`` property.  It supports
    the usual numpy and pandas iloc indexing semantics including
    integer indices, slices, lists and arrays of values. For more
    information see the ``Dataset.iloc`` property docstring.
    """

    @classmethod
    def _perform_getitem(cls, dataset, index):
        index = util.wrap_tuple(index)
        if len(index) == 1:
            index = (index[0], slice(None))
        elif len(index) > 2:
            raise IndexError('Tabular index not understood, index '
                             'must be at most length 2.')

        rows, cols = index
        if rows is Ellipsis:
            rows = slice(None)

        data = dataset.interface.iloc(dataset, (rows, cols))
        kdims = dataset.kdims
        vdims = dataset.vdims
        if util.isscalar(data):
            return data
        elif cols == slice(None):
            pass
        else:
            if isinstance(cols, slice):
                dims = dataset.dimensions()[index[1]]
            elif np.isscalar(cols):
                dims = [dataset.get_dimension(cols)]
            else:
                dims = [dataset.get_dimension(d) for d in cols]
            kdims = [d for d in dims if d in kdims]
            vdims = [d for d in dims if d in vdims]

        datatypes = util.unique_iterator([dataset.interface.datatype]+dataset.datatype)
        datatype = [dt for dt in datatypes if dt in Interface.interfaces and
                    not Interface.interfaces[dt].gridded]
        if not datatype: datatype = ['dataframe', 'dictionary']
        return dataset.clone(data, kdims=kdims, vdims=vdims, datatype=datatype)


class ndloc(Accessor):
    """
    ndloc is a small wrapper object that allows ndarray-like indexing
    for gridded Datasets using the ``.ndloc`` property. It supports
    the standard NumPy ndarray indexing semantics including
    integer indices, slices, lists and arrays of values. For more
    information see the ``Dataset.ndloc`` property docstring.
    """
    @classmethod
    def _perform_getitem(cls, dataset, indices):
        ds = dataset
        indices = util.wrap_tuple(indices)
        if not ds.interface.gridded:
            raise IndexError('Cannot use ndloc on non nd-dimensional datastructure')
        selected = dataset.interface.ndloc(ds, indices)
        if np.isscalar(selected):
            return selected
        params = {}
        if hasattr(ds, 'bounds'):
            params['bounds'] = None
        return dataset.clone(selected, datatype=[ds.interface.datatype]+ds.datatype, **params)


class Interface(param.Parameterized):

    interfaces = {}

    datatype = None

    types = ()

    # Denotes whether the interface expects gridded data
    gridded = False

    # Denotes whether the interface expects ragged data
    multi = False

    # Whether the interface stores the names of the underlying dimensions
    named = True

    ##################################################################
    # Initialization                                                 #
    ##################################################################

    @classmethod
    def loaded(cls):
        """
        Indicates whether the required dependencies are loaded.
        """
        return True

    @classmethod
    def applies(cls, obj):
        """
        Indicates whether the interface is designed specifically to
        handle the supplied object's type. By default simply checks
        if the object is one of the types declared on the class,
        however if the type is expensive to import at load time the
        method may be overridden.
        """
        return type(obj) in cls.types

    @classmethod
    def register(cls, interface):
        """"
        Registers a new Interface.
        """
        cls.interfaces[interface.datatype] = interface

    @classmethod
    def error(cls):
        """"
        Error message raised if interface could not resolve data.
        """
        info = dict(interface=cls.__name__)
        url = "http://holoviews.org/user_guide/%s_Datasets.html"
        if cls.multi:
            datatype = 'a list of tabular'
            info['url'] = url % 'Tabular'
        else:
            if cls.gridded:
                datatype = 'gridded'
            else:
                datatype = 'tabular'
            info['url'] = url % datatype.capitalize()
        info['datatype'] = datatype
        return ("{interface} expects {datatype} data, for more information "
                "on supported datatypes see {url}".format(**info))

    @classmethod
    def initialize(cls, eltype, data, kdims, vdims, datatype=None):
        # Process params and dimensions
        if isinstance(data, Element):
            pvals = util.get_param_values(data)
            kdims = pvals.get('kdims') if kdims is None else kdims
            vdims = pvals.get('vdims') if vdims is None else vdims

        # Process Element data
        if hasattr(data, 'interface') and isinstance(data.interface, type) and issubclass(data.interface, Interface):
            if datatype is None:
                datatype = [dt for dt in data.datatype if dt in eltype.datatype]
                if not datatype:
                    datatype = eltype.datatype

            interface = data.interface
            if interface.datatype in datatype and interface.datatype in eltype.datatype and interface.named:
                data = data.data
            elif interface.multi and any(cls.interfaces[dt].multi for dt in datatype if dt in cls.interfaces):
                data = [d for d in data.interface.split(data, None, None, 'columns')]
            elif interface.gridded and any(cls.interfaces[dt].gridded for dt in datatype):
                new_data = []
                for kd in data.kdims:
                    irregular = interface.irregular(data, kd)
                    coords = data.dimension_values(kd.name, expanded=irregular,
                                                   flat=not irregular)
                    new_data.append(coords)
                for vd in data.vdims:
                    new_data.append(interface.values(data, vd, flat=False, compute=False))
                data = tuple(new_data)
            elif 'dataframe' in datatype:
                data = data.dframe()
            else:
                data = tuple(data.columns().values())
        elif isinstance(data, Element):
            data = tuple(data.dimension_values(d) for d in kdims+vdims)
        elif isinstance(data, util.generator_types):
            data = list(data)

        if datatype is None:
            datatype = eltype.datatype

        # Set interface priority order
        prioritized = [cls.interfaces[p] for p in datatype
                       if p in cls.interfaces]
        head = [intfc for intfc in prioritized if intfc.applies(data)]
        if head:
            # Prioritize interfaces which have matching types
            prioritized = head + [el for el in prioritized if el != head[0]]

        # Iterate over interfaces until one can interpret the input
        priority_errors = []
        for interface in prioritized:
            if not interface.loaded() and len(datatype) != 1:
                # Skip interface if it is not loaded and was not explicitly requested
                continue
            try:
                (data, dims, extra_kws) = interface.init(eltype, data, kdims, vdims)
                break
            except DataError:
                raise
            except Exception as e:
                if interface in head or len(prioritized) == 1:
                    priority_errors.append((interface, e, True))
        else:
            error = ("None of the available storage backends were able "
                     "to support the supplied data format.")
            if priority_errors:
                intfc, e, _ = priority_errors[0]
                priority_error = f"{intfc.__name__} raised following error:\n\n {e}"
                error = ' '.join([error, priority_error])
                raise DataError(error, intfc).with_traceback(sys.exc_info()[2])
            raise DataError(error)

        return data, interface, dims, extra_kws

    @classmethod
    def validate(cls, dataset, vdims=True):
        """
        Validation runs after the Dataset has been constructed and should
        validate that the Dataset is correctly formed and contains
        all declared dimensions.
        """
        dims = 'all' if vdims else 'key'
        not_found = [d for d in dataset.dimensions(dims, label='name')
                     if d not in dataset.data]
        if not_found:
            raise DataError("Supplied data does not contain specified "
                            "dimensions, the following dimensions were "
                            "not found: %s" % repr(not_found), cls)

    ##################################################################
    # Utility methods                                                #
    ##################################################################

    @classmethod
    def cast(cls, datasets, datatype=None, cast_type=None):
        """
        Given a list of Dataset objects, cast them to the specified
        datatype (by default the format matching the current interface)
        with the given cast_type (if specified).
        """
        datatype = datatype or cls.datatype
        cast = []
        for ds in datasets:
            if cast_type is not None or ds.interface.datatype != datatype:
                ds = ds.clone(ds, datatype=[datatype], new_type=cast_type)
            cast.append(ds)
        return cast

    @classmethod
    def concatenate(cls, datasets, datatype=None, new_type=None):
        """
        Utility function to concatenate an NdMapping of Dataset objects.
        """
        from . import Dataset, default_datatype
        new_type = new_type or Dataset
        if isinstance(datasets, NdMapping):
            dimensions = datasets.kdims
            keys, datasets = zip(*datasets.data.items())
        elif isinstance(datasets, list) and all(not isinstance(v, tuple) for v in datasets):
            # Allow concatenating list of datasets (by declaring no dimensions and keys)
            dimensions, keys = [], [()]*len(datasets)
        else:
            raise DataError('Concatenation only supported for NdMappings '
                            'and lists of Datasets, found %s.' % type(datasets).__name__)

        template = datasets[0]
        datatype = datatype or template.interface.datatype

        # Handle non-general datatypes by casting to general type
        if datatype == 'array':
            datatype = default_datatype
        elif datatype == 'image':
            datatype = 'grid'

        if len(datasets) > 1 and not dimensions and cls.interfaces[datatype].gridded:
            raise DataError('Datasets with %s datatype cannot be concatenated '
                            'without defining the dimensions to concatenate along. '
                            'Ensure you pass in a NdMapping (e.g. a HoloMap) '
                            'of Dataset types, not a list.' % datatype)

        datasets = template.interface.cast(datasets, datatype)
        template = datasets[0]
        data = list(zip(keys, datasets)) if keys else datasets
        concat_data = template.interface.concat(data, dimensions, vdims=template.vdims)
        return template.clone(concat_data, kdims=dimensions+template.kdims, new_type=new_type)

    @classmethod
    def expanded(cls, arrays):
        return not any(array.shape not in [arrays[0].shape, (1,)] for array in arrays[1:])

    ##################################################################
    # Lazy and out-of-core data conversion                           #
    ##################################################################

    @classmethod
    def compute(cls, dataset):
        """
        Converts a lazy Dataset to a non-lazy, in-memory format.

        Note: This is a no-op if the data is already non-lazy.

        Args:
           dataset (Dataset)
        Returns:
           Dataset: Dataset with non-lazy data.
        """
        return dataset

    @classmethod
    def persist(cls, dataset):
        """
        Persists the data backing the Dataset in memory.

        Note: This is a no-op if the data is already in memory.

        Args:
           dataset (Dataset)
        Returns:
           Dataset: Dataset with the data persisted to memory.
        """
        return dataset

    ##################################################################
    # Shape info                                                     #
    ##################################################################

    @classmethod
    def length(cls, dataset):
        """
        Returns the number of rows in the Dataset

        Args:
           dataset (Dataset)
        Returns:
           int: Length of the data
        """
        return len(dataset.data)

    @classmethod
    def nonzero(cls, dataset):
        """
        Returns a boolean indicating whether the Dataset contains any data.

        Args:
           dataset (Dataset)
        Returns:
           bool: Whether the dataset is not empty.
        """
        return bool(cls.length(dataset))

    @classmethod
    def shape(cls, dataset):
        """
        Returns the shape of the data.

        Args:
           dataset (Dataset)
        Returns:
           tuple[int, int]: The shape of the data (rows, cols)
        """
        return dataset.data.shape

    ##################################################################
    # Dimension info                                                 #
    ##################################################################

    @classmethod
    def dtype(cls, dataset, dimension):
        """
        Returns the dtype for the selected dimension.

        Args:
           dataset (Dataset)
           dimension (str | Dimension): Dimension to return the dtype for.
        Returns:
           np.dtype: The dtype of the selected dimension.
        """
        name = dataset.get_dimension(dimension, strict=True).name
        data = dataset.data[name]
        if util.isscalar(data):
            return np.array([data]).dtype
        else:
            return data.dtype

    @classmethod
    def isscalar(cls, dataset, dimension):
        """
        Whether the selected dimension is a scalar value.

        Args:
           dataset (Dataset)
           dimension (str | Dimension): Dimension to check for scalar value.
        Returns:
           bool: Whether the dimension is scalar.
        """
        return len(cls.values(dataset, dimension, expanded=False)) == 1

    @classmethod
    def range(cls, dataset, dimension):
        """
        Computes the minimum and maximum value along a dimension.

        Note: In the past categorical and string columns were handled
              by sorting the values and taking the first and last value.
              This behavior is deprecated and will be removed in 2.0.
              In future the range for these columns will be returned
              as (None, None).

        Args:
           dataset (Dataset)
           dimension (str | Dimension): Dimension to compute the range on.
        Returns:
           Tuple[Any, Any]
        """
        column = dataset.dimension_values(dimension)
        if column.dtype.kind == 'M':
            return column.min(), column.max()
        elif len(column) == 0:
            return np.NaN, np.NaN
        else:
            try:
                assert column.dtype.kind not in 'SUO'
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                    return finite_range(column, np.nanmin(column), np.nanmax(column))
            except (AssertionError, TypeError):
                column = [v for v in util.python2sort(column) if v is not None]
                if not len(column):
                    return np.NaN, np.NaN
                return column[0], column[-1]

    @classmethod
    def values(cls, dataset, dimension, expanded=True, flat=True, compute=True, keep_index=False):
        """
        Returns the values along a dimension of the dataset.

        Note: The expanded keyword has different behavior for gridded
              interfaces where it determines whether 1D coordinates
              are expanded into a multi-dimensional array.

        Args:
           dataset (Dataset)
           dimension (str | Dimension): Dimension to return the values for.
           expanded (bool): When false returns unique values along the dimension.
           flat (bool): Whether to flatten the array.
           compute (bool): Whether to load lazy data into memory as a NumPy array.
           keep_index (bool): Whether to return the data with an index (if present).
        Returns:
           Dimension values in the requested format.
        """
        raise NotImplementedError

    ##################################################################
    # Adding dimensions                                              #
    ##################################################################

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        """
        Returns a copy of the data with the dimension values added.

        Args:
           dataset (Dataset): The Dataset to add the dimension to.
           dimension (Dimension): The dimension to add.
           dim_pos (int): The position in the data to add it to.
           values (array-like): The array of values to add.
           vdims (bool): Whether the data is a value dimension.
        Returns
           data: A copy of the data with the new dimension.
        """
        raise NotImplementedError

    @classmethod
    def assign(cls, dataset, new_data):
        """
        Adds a dictionary containing data for multiple new dimensions
        to a copy of the dataset.data.

        Args:
           dataset (Dataset): The Dataset to add the dimension to.
           new_data (Dict[str, array-like]): Dictionary containing new data to add to the Dataset.
        Returns:
           data: A copy of the data with the new data dimensions added.
        """
        raise NotImplementedError

    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        """

        """

        raise NotImplementedError

    ##################################################################
    # Selection utilities                                            #
    ##################################################################

    @classmethod
    def iloc(cls, dataset, index):
        """
        Implements integer indexing on the rows and columns of the data.

        Note: Only implement for tabular interfaces.

        Args:
            dataset (Dataset): The dataset to apply the indexing operation on.
            index (Tuple[Index, Index] | Index)
        """
        raise NotImplementedError

    @classmethod
    def select_mask(cls, dataset, selection):
        """
        Given a Dataset object and a dictionary with dimension keys and
        selection keys (i.e. tuple ranges, slices, sets, lists, or literals)
        return a boolean mask over the rows in the Dataset object that
        have been selected.

        Args:
           dataset (Dataset)
           selection (Dict[str, Any]): Dictionary containing selections for each column.
        Returns:
           np.ndarray[bool]: Boolean array representing the selectio mask.
        """
        mask = np.ones(len(dataset), dtype=np.bool_)
        for dim, sel in selection.items():
            if isinstance(sel, tuple):
                sel = slice(*sel)
            arr = cls.values(dataset, dim)
            if util.isdatetime(arr):
                try:
                    sel = util.parse_datetime_selection(sel)
                except Exception:
                    pass
            if isinstance(sel, slice):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'invalid value encountered')
                    if sel.start is not None:
                        mask &= sel.start <= arr
                    if sel.stop is not None:
                        mask &= arr < sel.stop
            elif isinstance(sel, (set, list)):
                iter_slcs = []
                for ik in sel:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', r'invalid value encountered')
                        iter_slcs.append(arr == ik)
                mask &= np.logical_or.reduce(iter_slcs)
            elif callable(sel):
                mask &= sel(arr)
            else:
                index_mask = arr == sel
                if dataset.ndims == 1 and np.sum(index_mask) == 0:
                    data_index = np.argmin(np.abs(arr - sel))
                    mask = np.zeros(len(dataset), dtype=np.bool_)
                    mask[data_index] = True
                else:
                    mask &= index_mask
        return mask

    @classmethod
    def indexed(cls, dataset, selection):
        """
        Given a Dataset object and selection to be applied returns
        boolean to indicate whether a scalar value has been indexed.
        """
        selected = list(selection.keys())
        all_scalar = all((not isinstance(sel, (tuple, slice, set, list))
                          and not callable(sel)) for sel in selection.values())
        all_kdims = all(d in selected for d in dataset.kdims)
        return all_scalar and all_kdims

    @classmethod
    def sample(cls, dataset, samples=[]):
        raise NotImplementedError

    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        raise NotImplementedError

    ##################################################################
    # Data transformations                                           #
    ##################################################################

    @classmethod
    def aggregate(cls, dataset, kdims, function, **kwargs):
        raise NotImplementedError

    @classmethod
    def histogram(cls, dimension_values, bins, density=True, weights=None):
        """
        Computes the histogram on the dimension values with support
        for specific bins, normalization and weighting.

        Note: Usually the dimension_values and weights are assumed to
              be arrays but each interface should support data stored
              in whatever format it uses to store dimensions internally.

        Args:
           dimension_values (array-like): In memory representation of the dimension values.
           bins (np.ndarray | int): An array of bins or the number of bins.
           density (bool): Whether to normalize the histogram.
           weights (array-like): In memory representation of the weighting.
        Returns:
           Tuple[array-like, array-like]: Tuple of histogram values and bin edges.
        """
        if util.is_dask_array(dimension_values):
            import dask.array as da
            histogram = da.histogram
        elif util.is_cupy_array(dimension_values):
            import cupy
            histogram = cupy.histogram
        else:
            histogram = np.histogram
        hist, edges = histogram(dimension_values, bins=bins, density=density, weights=weights)
        if util.is_cupy_array(hist):
            edges = cupy.asnumpy(edges)
            hist = cupy.asnumpy(hist)
        return hist, edges

    @classmethod
    def mask(cls, dataset, mask, mask_value=np.nan):
        raise NotImplementedError

    @classmethod
    def redim(cls, dataset, dimensions):
        """
        Renames dimensions in the data.

        Note: Only meaningful for data formats that store dimension names.

        Args:
           dataset (Dataset)
           dimensions (Dict[str, str]): Dictionary mapping from old to new dimension names.
        Returns:
           Data after the dimension names have been transformed.
        """
        return dataset.data

    @classmethod
    def reindex(cls, dataset, kdims, vdims):
        """
        Reindexes data given new key and value dimensions.
        """
        raise NotImplementedError

    @classmethod
    def reduce(cls, dataset, reduce_dims, function, **kwargs):
        """
        Reduces one or more dimensions using the supplied reduction function.

        Args:
           dataset (Dataset)
           dimensions (List[str | Dimension]): List of dimensions to reduce.
           function (str | np.ufunc): Reduction operation to apply.
        Returns:
           Dataset: Dataset containing the reduced (or aggregated) data.
        """
        kdims = [kdim for kdim in dataset.kdims if kdim not in reduce_dims]
        return cls.aggregate(dataset, kdims, function, **kwargs)

    @classmethod
    def replace_value(cls, data, nodata):
        """
        Replace `nodata` value in data with NaN

        Args:
           dataset (Dataset)
           nodata (numbers.Number): The nodata value to replace.
        Returns:
           np.ndarray: Array with the nodata value replaced with NaN.
        """
        data = data.astype('float64')
        mask = data != nodata
        if hasattr(data, 'where'):
            return data.where(mask, np.NaN)
        return np.where(mask, data, np.NaN)

    @classmethod
    def sort(cls, dataset, by=[], reverse=False):
        raise NotImplementedError

    ##################################################################
    # Conversion methods                                             #
    ##################################################################

    @classmethod
    def as_dframe(cls, dataset):
        """
        Returns the data of a Dataset as a dataframe avoiding copying
        if it already a dataframe type.
        """
        return dataset.dframe()

    @classmethod
    def array(cls, dataset, dimensions):
        """
        Returns the data as a numpy.ndarray containing the selected dimensions.

        Args:
           dataset (Dataset)
           dimensions (List[str]): List of dimensions to include.
        Returns:
           np.ndarray: A Numpy ndarray containing the selected dimensions.
        """
        return Element.array(dataset, dimensions)

    @classmethod
    def dframe(cls, dataset, dimensions):
        """
        Returns the data as a pandas.DataFrame containing the selected dimensions.

        Args:
           dataset (Dataset)
           dimensions (List[str]): List of dimensions to include.
        Returns:
           pandas.DataFrame: A DataFrame containing the selected dimensions.
        """
        return Element.dframe(dataset, dimensions)

    @classmethod
    def columns(cls, dataset, dimensions):
        """
        Returns the data as a dictionary of 1D arrays indexed by column name.

        Args:
           dataset (Dataset)
           dimensions (List[str]): List of dimensions to include.
        Returns:
           Dict[str, np.ndarray]: Whether the dataset is not empty.
        """
        return Element.columns(dataset, dimensions)

    ##################################################################
    # Geometry methods                                               #
    ##################################################################

    @classmethod
    def geom_type(cls, dataset):
        raise NotImplementedError

    @classmethod
    def has_holes(cls, dataset):
        """
        Whether the Dataset contains geometries with holes.

        Note: Only meaningful to implement on Interfaces that support
              geometry data.

        Args:
           dataset (Dataset)
        Returns:
           bool: Whether the Dataset contains geometries with holes.
        """
        return False

    @classmethod
    def holes(cls, dataset):
        """
        Returns a list of lists of arrays containing the holes for
        each geometry in the Dataset.

        Note: Only meaningful to implement on Interfaces that support
              geometry data.

        Args:
           dataset (Dataset)
        Returns:
           List[List[np.ndarray]]: List of list of arrays representing geometry holes.
        """
        coords = cls.values(dataset, dataset.kdims[0])
        splits = np.where(np.isnan(coords.astype('float')))[0]
        return [[[]]*(len(splits)+1)]

    @classmethod
    def isunique(cls, dataset, dim, per_geom=False):
        """
        Whether the selected dimension has only a single unique value.

        Compatibility method introduced for v1.13.0 to smooth
        over addition of per_geom kwarg for isscalar method.

        Args:
           dataset (Dataset)
           dimension (str | Dimension): Dimension to check for scalar value.
           per_geom (bool): If geometry
        Returns:
           bool: Whether the dimension is scalar.
        """
        try:
            return cls.isscalar(dataset, dim, per_geom)
        except TypeError:
            return cls.isscalar(dataset, dim)
