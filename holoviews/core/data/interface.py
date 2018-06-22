import warnings

import param
import numpy as np

from ..element import Element
from ..ndmapping import OrderedDict, NdMapping
from .. import util


class DataError(ValueError):
    "DataError is raised when the data cannot be interpreted"

    def __init__(self, msg, interface=None):
        if interface is not None:
            msg = '\n\n'.join([msg, interface.error()])
        super(DataError, self).__init__(msg)


class iloc(object):
    """
    iloc is small wrapper object that allows row, column based
    indexing into a Dataset using the ``.iloc`` property.  It supports
    the usual numpy and pandas iloc indexing semantics including
    integer indices, slices, lists and arrays of values. For more
    information see the ``Dataset.iloc`` property docstring.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        index = util.wrap_tuple(index)
        if len(index) == 1:
            index = (index[0], slice(None))
        elif len(index) > 2:
            raise IndexError('Tabular index not understood, index '
                             'must be at most length 2.')

        rows, cols = index
        if rows is Ellipsis:
            rows = slice(None)
        data = self.dataset.interface.iloc(self.dataset, (rows, cols))
        kdims = self.dataset.kdims
        vdims = self.dataset.vdims
        if np.isscalar(data):
            return data
        elif cols == slice(None):
            pass
        else:
            if isinstance(cols, slice):
                dims = self.dataset.dimensions()[index[1]]
            elif np.isscalar(cols):
                dims = [self.dataset.get_dimension(cols)]
            else:
                dims = [self.dataset.get_dimension(d) for d in cols]
            kdims = [d for d in dims if d in kdims]
            vdims = [d for d in dims if d in vdims]

        datatype = [dt for dt in self.dataset.datatype
                    if dt in Interface.interfaces and
                    not Interface.interfaces[dt].gridded]
        if not datatype: datatype = ['dataframe', 'dictionary']
        return self.dataset.clone(data, kdims=kdims, vdims=vdims,
                                  datatype=datatype)


class ndloc(object):
    """
    ndloc is a small wrapper object that allows ndarray-like indexing
    for gridded Datasets using the ``.ndloc`` property. It supports
    the standard NumPy ndarray indexing semantics including
    integer indices, slices, lists and arrays of values. For more
    information see the ``Dataset.ndloc`` property docstring.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, indices):
        ds = self.dataset
        indices = util.wrap_tuple(indices)
        if not ds.interface.gridded:
            raise IndexError('Cannot use ndloc on non nd-dimensional datastructure')
        selected = self.dataset.interface.ndloc(ds, indices)
        if np.isscalar(selected):
            return selected
        params = {}
        if hasattr(ds, 'bounds'):
            params['bounds'] = None
        return self.dataset.clone(selected, datatype=[ds.interface.datatype]+ds.datatype, **params)


class Interface(param.Parameterized):

    interfaces = {}

    datatype = None

    # Denotes whether the interface expects gridded data
    gridded = False

    # Denotes whether the interface expects ragged data
    multi = False

    @classmethod
    def register(cls, interface):
        cls.interfaces[interface.datatype] = interface

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
    def error(cls):
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
        if (hasattr(data, 'interface') and issubclass(data.interface, Interface)):
            if datatype is None:
                datatype = [dt for dt in data.datatype if dt in eltype.datatype]
                if not datatype:
                    datatype = eltype.datatype

            if data.interface.datatype in datatype and data.interface.datatype in eltype.datatype:
                data = data.data
            elif data.interface.gridded and any(cls.interfaces[dt].gridded for dt in datatype):
                gridded = OrderedDict([(kd.name, data.dimension_values(kd.name, expanded=False))
                                       for kd in data.kdims])
                for vd in data.vdims:
                    gridded[vd.name] = data.dimension_values(vd, flat=False)
                data = tuple(gridded.values())
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
        head = [intfc for intfc in prioritized if type(data) in intfc.types]
        if head:
            # Prioritize interfaces which have matching types
            prioritized = head + [el for el in prioritized if el != head[0]]

        # Iterate over interfaces until one can interpret the input
        priority_errors = []
        for interface in prioritized:
            try:
                (data, dims, extra_kws) = interface.init(eltype, data, kdims, vdims)
                break
            except DataError:
                raise
            except Exception as e:
                if interface in head:
                    priority_errors.append((interface, e))
        else:
            error = ("None of the available storage backends were able "
                     "to support the supplied data format.")
            if priority_errors:
                intfc, e = priority_errors[0]
                priority_error = ("%s raised following error:\n\n %s"
                                  % (intfc.__name__, e))
                error = ' '.join([error, priority_error])
            raise DataError(error)

        return data, interface, dims, extra_kws


    @classmethod
    def validate(cls, dataset, vdims=True):
        dims = 'all' if vdims else 'key'
        not_found = [d for d in dataset.dimensions(dims, label='name')
                     if d not in dataset.data]
        if not_found:
            raise DataError("Supplied data does not contain specified "
                            "dimensions, the following dimensions were "
                            "not found: %s" % repr(not_found), cls)


    @classmethod
    def expanded(cls, arrays):
        return not any(array.shape not in [arrays[0].shape, (1,)] for array in arrays[1:])


    @classmethod
    def isscalar(cls, dataset, dim):
        return cls.values(dataset, dim, expanded=False) == 1


    @classmethod
    def select_mask(cls, dataset, selection):
        """
        Given a Dataset object and a dictionary with dimension keys and
        selection keys (i.e tuple ranges, slices, sets, lists or literals)
        return a boolean mask over the rows in the Dataset object that
        have been selected.
        """
        mask = np.ones(len(dataset), dtype=np.bool)
        for dim, k in selection.items():
            if isinstance(k, tuple):
                k = slice(*k)
            arr = cls.values(dataset, dim)
            if isinstance(k, slice):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'invalid value encountered')
                    if k.start is not None:
                        mask &= k.start <= arr
                    if k.stop is not None:
                        mask &= arr < k.stop
            elif isinstance(k, (set, list)):
                iter_slcs = []
                for ik in k:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', r'invalid value encountered')
                        iter_slcs.append(arr == ik)
                mask &= np.logical_or.reduce(iter_slcs)
            elif callable(k):
                mask &= k(arr)
            else:
                index_mask = arr == k
                if dataset.ndims == 1 and np.sum(index_mask) == 0:
                    data_index = np.argmin(np.abs(arr - k))
                    mask = np.zeros(len(dataset), dtype=np.bool)
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
    def range(cls, dataset, dimension):
        column = dataset.dimension_values(dimension)
        if column.dtype.kind == 'M':
            return column.min(), column.max()
        elif len(column) == 0:
            return np.NaN, np.NaN
        else:
            try:
                assert column.dtype.kind not in 'SUO'
                return (np.nanmin(column), np.nanmax(column))
            except (AssertionError, TypeError):
                column = [v for v in util.python2sort(column) if v is not None]
                if not len(column):
                    return np.NaN, np.NaN
                return column[0], column[-1]

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
    def reduce(cls, dataset, reduce_dims, function, **kwargs):
        kdims = [kdim for kdim in dataset.kdims if kdim not in reduce_dims]
        return cls.aggregate(dataset, kdims, function, **kwargs)

    @classmethod
    def array(cls, dataset, dimensions):
        return Element.array(dataset, dimensions)

    @classmethod
    def dframe(cls, dataset, dimensions):
        return Element.dframe(dataset, dimensions)

    @classmethod
    def columns(cls, dataset, dimensions):
        return Element.columns(dataset, dimensions)

    @classmethod
    def shape(cls, dataset):
        return dataset.data.shape

    @classmethod
    def length(cls, dataset):
        return len(dataset.data)

    @classmethod
    def nonzero(cls, dataset):
        return bool(cls.length(dataset))

    @classmethod
    def redim(cls, dataset, dimensions):
        return dataset.data
