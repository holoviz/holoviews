try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np

from .interface import Interface
from ..dimension import Dimension, Dimensioned
from ..element import NdElement
from ..ndmapping import item_check
from .. import util


class NdElementInterface(Interface):

    types = (NdElement,)

    datatype = 'ndelement'

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        if isinstance(data, NdElement):
            kdims = [d for d in kdims if d != 'Index']
        else:
            element_params = eltype.params()
            kdims = kdims if kdims else element_params['kdims'].default
            vdims = vdims if vdims else element_params['vdims'].default

        dimensions = [d.name if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if ((isinstance(data, dict) or util.is_dataframe(data)) and
            all(d in data for d in dimensions)):
            data = tuple(data.get(d) for d in dimensions)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                if eltype._1d:
                    data = np.atleast_2d(data).T
                else:
                    data = (np.arange(len(data)), data)
            else:
                data = tuple(data[:, i]  for i in range(data.shape[1]))
        elif isinstance(data, list) and np.isscalar(data[0]):
            data = (np.arange(len(data)), data)

        if not isinstance(data, (NdElement, dict)):
            # If ndim > 2 data is assumed to be a mapping

            if (isinstance(data[0], tuple) and any(isinstance(d, tuple) for d in data[0])):
                pass
            else:
                if isinstance(data, tuple):
                    data = tuple(np.array(d) if not isinstance(d, np.ndarray) else d for d in data)
                    if not cls.expanded(data):
                        raise ValueError('NdElementInterface expects data to be of uniform shape')
                    data = zip(*data)
                ndims = len(kdims)
                data = [(tuple(row[:ndims]), tuple(row[ndims:]))
                        for row in data]
        if isinstance(data, (dict, list)):
            data = NdElement(data, kdims=kdims, vdims=vdims)
        elif not isinstance(data, NdElement):
            raise ValueError("NdElementInterface interface couldn't convert data.""")
        return data, {'kdims':kdims, 'vdims':vdims}, {}


    @classmethod
    def validate(cls, columns):
        """
        NdElement will validate the data
        """
        pass

    @classmethod
    def dimension_type(cls, columns, dim):
        return Dimensioned.get_dimension_type(columns, dim)

    @classmethod
    def shape(cls, columns):
        return (len(columns), len(columns.dimensions()))

    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values, vdim):
        return columns.data.add_dimension(dimension, dim_pos+1, values, vdim)

    @classmethod
    def concat(cls, columns_objs):
        cast_objs = cls.cast(columns_objs)
        return [(k[1:], v) for col in cast_objs for k, v in col.data.data.items()]

    @classmethod
    def sort(cls, columns, by=[]):
        if not len(by): by = columns.dimensions('key', True)
        return columns.data.sort(by)

    @classmethod
    def values(cls, columns, dim, expanded=True, flat=True):
        values = columns.data.dimension_values(dim, expanded, flat)
        if not expanded:
            return util.unique_array(values)
        return values

    @classmethod
    def reindex(cls, columns, kdims=None, vdims=None):
        return columns.data.reindex(kdims, vdims)

    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        if 'kdims' not in kwargs:
            kwargs['kdims'] = [d for d in columns.kdims if d not in dimensions]
        with item_check(False):
            return columns.data.groupby(dimensions, container_type, group_type, **kwargs)

    @classmethod
    def select(cls, columns, selection_mask=None, **selection):
        if selection_mask is None:
            return columns.data.select(**selection)
        else:
            return columns.data[selection_mask]

    @classmethod
    def sample(cls, columns, samples=[]):
        return columns.data.sample(samples)

    @classmethod
    def reduce(cls, columns, reduce_dims, function, **kwargs):
        return columns.data.reduce(columns.data, reduce_dims, function)

    @classmethod
    def aggregate(cls, columns, dimensions, function, **kwargs):
        return columns.data.aggregate(dimensions, function, **kwargs)

    @classmethod
    def unpack_scalar(cls, columns, data):
        if len(data) == 1 and len(data.kdims) == 1 and len(data.vdims) == 1:
            return list(data.data.values())[0][0]
        else:
            return data


Interface.register(NdElementInterface)
