import numpy as np

from ..util import max_range
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
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            ds.interface.validate(ds, vdims)


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
        return max_range(ranges)


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
        if not self.dataset.data:
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
    def aggregate(cls, columns, dimensions, function, **kwargs):
        raise NotImplementedError('Aggregation currently not implemented')

    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        raise NotImplementedError('Grouping currently not implemented')

    @classmethod
    def sample(cls, columns, samples=[]):
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
        didx = dataset.get_dimension_index(dimension)
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
            return np.array()
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


Interface.register(MultiInterface)
