import numpy as np

from .. import util
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .dictionary import DictInterface
from .interface import Interface, DataError


class MultiInterface(Interface):
    """
    MultiInterface allows wrapping around a list of tabular datasets
    including dataframes, the columnar dictionary format or 2D tabular
    NumPy arrays. Using the split method the list of tabular data can
    be split into individual datasets.

    The interface makes the data appear a list of tabular datasets as
    a single dataset. The length, shape and values methods therefore
    make the data appear like a single array of concatenated subpaths,
    separated by NaN values.
    """

    types = ()

    datatype = 'multitabular'

    subtypes = ['dictionary', 'dataframe', 'array', 'dask']

    multi = True

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        new_data = []
        dims = {'kdims': eltype.kdims, 'vdims': eltype.vdims}
        if kdims is not None:
            dims['kdims'] = kdims
        if vdims is not None:
            dims['vdims'] = vdims
        if not isinstance(data, list):
            raise ValueError('MultiInterface data must be a list tabular data types.')
        prev_interface, prev_dims = None, None
        for d in data:
            d, interface, dims, _ = Interface.initialize(eltype, d, kdims, vdims,
                                                         datatype=cls.subtypes)
            if prev_interface:
                if prev_interface != interface:
                    raise DataError('MultiInterface subpaths must all have matching datatype.', cls)
                if dims['kdims'] != prev_dims['kdims']:
                    raise DataError('MultiInterface subpaths must all have matching kdims.', cls)
                if dims['vdims'] != prev_dims['vdims']:
                    raise DataError('MultiInterface subpaths must all have matching vdims.', cls)
            new_data.append(d)
            prev_interface, prev_dims = interface, dims
        return new_data, dims, {}

    @classmethod
    def validate(cls, dataset, vdims=True):
        if not dataset.data:
            return

        from holoviews.element import Polygons
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            ds.interface.validate(ds, vdims)
            if isinstance(dataset, Polygons) and ds.interface is DictInterface:
                holes = ds.interface.holes(ds)
                if not isinstance(holes, list):
                    raise DataError('Polygons holes must be declared as a list-of-lists.', cls)
                subholes = holes[0]
                coords = ds.data[ds.kdims[0].name]
                splits = np.isnan(coords.astype('float')).sum()
                if len(subholes) != (splits+1):
                    raise DataError('Polygons with holes containing multi-geometries '
                                    'must declare a list of holes for each geometry.', cls)


    @classmethod
    def _inner_dataset_template(cls, dataset):
        """
        Returns a Dataset template used as a wrapper around the data
        contained within the multi-interface dataset.
        """
        from . import Dataset
        vdims = dataset.vdims if getattr(dataset, 'level', None) is None else []
        return Dataset(dataset.data[0], datatype=cls.subtypes,
                       kdims=dataset.kdims, vdims=vdims)

    @classmethod
    def dimension_type(cls, dataset, dim):
        if not dataset.data:
            # Note: Required to make empty datasets work at all (should fix)
            # Other interfaces declare equivalent of empty array
            # which defaults to float type
            return float
        ds = cls._inner_dataset_template(dataset)
        return ds.interface.dimension_type(ds, dim)

    @classmethod
    def range(cls, dataset, dim):
        if not dataset.data:
            return (None, None)
        ranges = []
        ds = cls._inner_dataset_template(dataset)

        # Backward compatibility for Contours/Polygons level
        level = getattr(dataset, 'level', None)
        dim = dataset.get_dimension(dim)
        if level is not None and dim is dataset.vdims[0]:
            return (level, level)

        for d in dataset.data:
            ds.data = d
            ranges.append(ds.interface.range(ds, dim))
        return util.max_range(ranges)


    @classmethod
    def has_holes(cls, dataset):
        if not dataset.data:
            return False
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            if ds.interface.has_holes(ds):
                return True
        return False

    @classmethod
    def holes(cls, dataset):
        holes = []
        if not dataset.data:
            return holes
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            holes += ds.interface.holes(ds)
        return holes


    @classmethod
    def isscalar(cls, dataset, dim):
        """
        Tests if dimension is scalar in each subpath.
        """
        if not dataset.data:
            return True
        ds = cls._inner_dataset_template(dataset)
        isscalar = []
        for d in dataset.data:
            ds.data = d
            isscalar.append(ds.interface.isscalar(ds, dim))
        return all(isscalar)


    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        """
        Applies selectiong on all the subpaths.
        """
        if not dataset.data:
            return []
        ds = cls._inner_dataset_template(dataset)
        data = []
        for d in dataset.data:
            ds.data = d
            sel = ds.interface.select(ds, **selection)
            data.append(sel)
        return data

    @classmethod
    def select_paths(cls, dataset, selection):
        """
        Allows selecting paths with usual NumPy slicing index.
        """
        return [s[0] for s in np.array([{0: p} for p in dataset.data])[selection]]

    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        raise NotImplementedError('Aggregation currently not implemented')

    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        # Get dimensions information
        dimensions = [dataset.get_dimension(d) for d in dimensions]
        kdims = [kdim for kdim in dataset.kdims if kdim not in dimensions]

        # Update the kwargs appropriately for Element group types
        group_kwargs = {}
        group_type = list if group_type == 'raw' else group_type
        if issubclass(group_type, Element):
            group_kwargs.update(util.get_param_values(dataset))
            group_kwargs['kdims'] = kdims
        group_kwargs.update(kwargs)

        # Find all the keys along supplied dimensions
        values = []
        for d in dimensions:
            if not cls.isscalar(dataset, d):
                raise ValueError('MultiInterface can only apply groupby '
                                 'on scalar dimensions, %s dimension'
                                 'is not scalar' % d)
            vals = cls.values(dataset, d, False, True)
            values.append(vals)
        values = tuple(values)

        # Iterate over the unique entries applying selection masks
        from . import Dataset
        ds = Dataset(values, dimensions)
        keys = (tuple(vals[i] for vals in values) for i in range(len(vals)))
        grouped_data = []
        for unique_key in util.unique_iterator(keys):
            mask = ds.interface.select_mask(ds, dict(zip(dimensions, unique_key)))
            selection = [data for data, m in zip(dataset.data, mask) if m]
            group_data = group_type(selection, **group_kwargs)
            grouped_data.append((unique_key, group_data))

        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(grouped_data, kdims=dimensions)
        else:
            return container_type(grouped_data)

    @classmethod
    def sample(cls, dataset, samples=[]):
        raise NotImplementedError('Sampling operation on subpaths not supported')

    @classmethod
    def shape(cls, dataset):
        """
        Returns the shape of all subpaths, making it appear like a
        single array of concatenated subpaths separated by NaN values.
        """
        if not dataset.data:
            return (0, len(dataset.dimensions()))

        rows, cols = 0, 0
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            r, cols = ds.interface.shape(ds)
            rows += r
        return rows+len(dataset.data)-1, cols

    @classmethod
    def length(cls, dataset):
        """
        Returns the length of the multi-tabular dataset making it appear
        like a single array of concatenated subpaths separated by NaN
        values.
        """
        if not dataset.data:
            return 0
        length = 0
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            length += ds.interface.length(ds)
        return length+len(dataset.data)-1

    @classmethod
    def nonzero(cls, dataset):
        return bool(dataset.data)

    @classmethod
    def redim(cls, dataset, dimensions):
        if not dataset.data:
            return dataset.data
        new_data = []
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            new_data.append(ds.interface.redim(ds, dimensions))
        return new_data

    @classmethod
    def values(cls, dataset, dimension, expanded, flat):
        """
        Returns a single concatenated array of all subpaths separated
        by NaN values. If expanded keyword is False an array of arrays
        is returned.
        """
        if not dataset.data:
            return np.array([])
        values = []
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            dvals = ds.interface.values(ds, dimension, expanded, flat)
            if not len(dvals):
                continue
            elif expanded:
                values.append(dvals)
                values.append([np.NaN])
            else:
                values.append(dvals)
        if not values:
            return np.array([])
        elif expanded:
            return np.concatenate(values[:-1])
        else:
            return np.concatenate(values)

    @classmethod
    def split(cls, dataset, start, end, datatype, **kwargs):
        """
        Splits a multi-interface Dataset into regular Datasets using
        regular tabular interfaces.
        """
        objs = []
        if datatype is None:
            for d in dataset.data[start: end]:
                objs.append(dataset.clone(d, datatype=cls.subtypes))
            return objs
        elif not dataset.data:
            return objs
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            if datatype == 'array':
                obj = ds.array(**kwargs)
            elif datatype == 'dataframe':
                obj = ds.dframe(**kwargs)
            elif datatype == 'columns':
                if ds.interface.datatype == 'dictionary':
                    obj = dict(d)
                else:
                    obj = ds.columns(**kwargs)
            else:
                raise ValueError("%s datatype not support" % datatype)
            objs.append(obj)
        return objs

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        if not len(dataset.data):
            return dataset.data
        elif values is None or util.isscalar(values):
            values = [values]*len(dataset.data)
        elif not len(values) == len(dataset.data):
            raise ValueError('Added dimension values must be scalar or '
                             'match the length of the data.')

        new_data = []
        template = cls._inner_dataset_template(dataset)
        array_type = template.interface.datatype == 'array'
        for d, v in zip(dataset.data, values):
            template.data = d
            if array_type:
                ds = template.clone(template.columns())
            else:
                ds = template
            new_data.append(ds.interface.add_dimension(ds, dimension, dim_pos, v, vdim))
        return new_data



Interface.register(MultiInterface)
