import numpy as np

from ..util import max_range
from .interface import Interface

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

    subtypes = ['dataframe', 'dictionary', 'array', 'dask']

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        new_data = []
        dims = {'kdims': eltype.kdims, 'vdims': eltype.vdims}
        if not isinstance(data, list):
            raise ValueError('MultiInterface data must be a list tabular data types.')
        prev_interface, prev_dims = None, None
        for d in data:
            d, interface, dims, _ = Interface.initialize(eltype, d, kdims, vdims,
                                                         datatype=cls.subtypes)
            if prev_interface:
                if prev_interface != interface:
                    raise ValueError('MultiInterface subpaths must all have matching datatype.')
                if dims['kdims'] != prev_dims['kdims']:
                    raise ValueError('MultiInterface subpaths must all have matching kdims.')
                if dims['vdims'] != prev_dims['vdims']:
                    raise ValueError('MultiInterface subpaths must all have matching vdims.')
            new_data.append(d)
            prev_interface, prev_dims = interface, dims
        return new_data, dims, {}

    @classmethod
    def validate(cls, dataset):
        # Ensure that auxilliary key dimensions on each subpaths are scalar
        if dataset.ndims <= 2:
            return
        ds = cls._inner_dataset_template(dataset)
        for d in dataset.data:
            ds.data = d
            for dim in dataset.kdims[2:]:
                if len(ds.dimension_values(dim, expanded=False)) > 1:
                    raise ValueError("'%s' key dimension value must have a constant value on each subpath, "
                                     "for paths with value for each coordinate of the array declare a "
                                     "value dimension instead." % dim)

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
    def select(cls, dataset, selection_mask=None, **selection):
        """
        Applies selectiong on all the subpaths.
        """
        ds = cls._inner_dataset_template(dataset)
        data = []
        for d in dataset.data:
            ds.data = d
            sel = ds.interface.select(ds, **selection)
            data.append(sel)
        return data

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
        return bool(cls.length(dataset))

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
            expand = expanded if didx>1 and dimension in dataset.kdims else True
            dvals = ds.interface.values(ds, dimension, expand, flat)
            values.append(dvals)
            if expanded:
                values.append([np.NaN])
            elif not expand and len(dvals):
                values[-1] = dvals[0]
        if not values:
            return np.array()
        elif expanded:
            return np.concatenate(values[:-1])
        else:
            return np.array(values)

    @classmethod
    def split(cls, dataset, start, end):
        """
        Splits a multi-interface Dataset into regular Datasets using
        regular tabular interfaces.
        """
        objs = []
        for d in dataset.data[start: end]:
            objs.append(dataset.clone(d, datatype=cls.subtypes))
        return objs


Interface.register(MultiInterface)
