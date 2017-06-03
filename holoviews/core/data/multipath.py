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
        for d in data:
            d, interface, dims, extra_kws = Interface.initialize(eltype, d, kdims, vdims,
                                                                 datatype=cls.subtypes)
            new_data.append(d)
        return new_data, dims, extra_kws

    @classmethod
    def validate(cls, dataset):
        pass

    @classmethod
    def dimension_type(cls, dataset, dim):
        ds = dataset.clone(dataset.data[0], datatype=cls.subtypes, vdims=[])
        return ds.interface.dimension_type(ds, dim)

    @classmethod
    def range(cls, dataset, dim):
        ranges = []
        ds = dataset.clone(dataset.data[0], datatype=cls.subtypes,
                           vdims=[])
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
        rows, cols = 0, 0
        ds = dataset.clone(dataset.data[0], datatype=cls.subtypes,
                           vdims=[])
        for d in dataset.data:
            ds.data = d
            r, cols = ds.interface.shape(ds)
            rows += r
        return rows+len(dataset.data)-1, cols

    @classmethod
    def length(cls, dataset):
        length = 0
        ds = dataset.clone(dataset.data[0], datatype=cls.subtypes,
                           vdims=[])
        for d in dataset.data:
            ds.data = d
            length += ds.interface.length(ds)
        return length+len(dataset.data)-1

    @classmethod
    def nonzero(cls, dataset):
        return bool(cls.length(dataset))

    @classmethod
    def redim(cls, dataset, dimensions):
        new_data = []
        ds = dataset.clone(dataset.data[0], datatype=cls.subtypes,
                           vdims=[])
        for d in dataset.data:
            ds.data = d
            new_data.append(ds.interface.redim(ds, dimensions))
        return new_data

    @classmethod
    def values(cls, dataset, dimension, expanded, flat):
        values = []
        ds = dataset.clone(dataset.data[0], datatype=cls.subtypes,
                           vdims=[])
        for d in dataset.data:
            ds.data = d
            values.append(ds.interface.values(ds, dimension))
            values.append([np.NaN])
        return np.concatenate(values[:-1]) if values else []

    @classmethod
    def split(cls, dataset, start, end):
        objs = []
        for d in dataset.data[start: end]:
            objs.append(dataset.clone(d, datatype=cls.subtypes, vdims=[]))
        return objs


Interface.register(MultiInterface)
