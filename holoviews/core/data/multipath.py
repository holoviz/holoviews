import numpy as np

from ..util import max_range
from .interface import Interface

class MultiInterface(Interface):

    types = (list,)

    datatype = 'multi'

    subtypes = ['dataframe', 'dictionary', 'array']

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        new_data = []
        dims = {'kdims': eltype.kdims, 'vdims': eltype.vdims}
        extra_kws = {}
        for d in data:
            d, interface, dims, extra_kws = Interface.initialize(eltype, d, kdims, vdims,
                                                                 datatype=cls.subtypes)
            new_data.append(d)
        return new_data, dims, extra_kws

    @classmethod
    def validate(cls, dataset):
        pass


    @classmethod
    def template(cls, dataset):
        from . import Dataset
        vdims = dataset.vdims if getattr(dataset, 'level', None) is None else []
        return dataset.clone(dataset.data[0], datatype=cls.subtypes,
                             vdims=vdims, new_type=Dataset)


    @classmethod
    def dimension_type(cls, dataset, dim):
        if not dataset.data:
            return float
        ds = cls.template(dataset)
        return ds.interface.dimension_type(ds, dim)

    @classmethod
    def range(cls, dataset, dim):
        if not dataset.data:
            return (None, None)
        ranges = []
        ds = cls.template(dataset)

        # Backward compatibility for level
        level = getattr(dataset, 'level', None)
        dim = dataset.get_dimension(dim)
        if level is not None and dim is dataset.vdims[0]:
            return (level, level)

        for d in dataset.data:
            ds.data = d
            ranges.append(ds.interface.range(ds, dim))
        return max_range(ranges)

    @classmethod
    def aggregate(cls, columns, dimensions, function, **kwargs):
        raise NotImplementedError

    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        raise NotImplementedError

    @classmethod
    def sample(cls, columns, samples=[]):
        raise NotImplementedError

    @classmethod
    def shape(cls, dataset):
        if not dataset.data:
            return (0, len(dataset.dimensions()))

        rows, cols = 0, 0
        ds = cls.template(dataset)
        for d in dataset.data:
            ds.data = d
            r, cols = ds.interface.shape(ds)
            rows += r
        return rows+len(dataset.data)-1, cols

    @classmethod
    def length(cls, dataset):
        if not dataset.data:
            return 0
        length = 0
        ds = cls.template(dataset)
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
        ds = cls.template(dataset)
        for d in dataset.data:
            ds.data = d
            new_data.append(ds.interface.redim(ds, dimensions))
        return new_data

    @classmethod
    def values(cls, dataset, dimension, expanded, flat):
        if not dataset.data:
            return np.array([])
        values = []
        ds = cls.template(dataset)
        for d in dataset.data:
            ds.data = d
            values.append(ds.interface.values(ds, dimension, expanded, flat))
            if expanded:
                values.append([np.NaN])
        return np.concatenate(values[:-1] if expanded else values) if values else []

    @classmethod
    def split(cls, dataset, start, end):
        objs = []
        for d in dataset.data[start: end]:
            objs.append(dataset.clone(d, datatype=cls.subtypes))
        return objs


Interface.register(MultiInterface)
