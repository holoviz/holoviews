from __future__ import absolute_import

from distutils.version import LooseVersion

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np
import pandas as pd

from .interface import Interface
from ..dimension import Dimension
from ..element import Element, NdElement
from ..dimension import OrderedDict as cyODict
from ..ndmapping import NdMapping, item_check
from .. import util


class PandasInterface(Interface):

    types = (pd.DataFrame if pd else None,)

    datatype = 'dataframe'

    @classmethod
    def dimension_type(cls, columns, dim):
        name = columns.get_dimension(dim).name
        idx = list(columns.data.columns).index(name)
        return columns.data.dtypes[idx].type

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        element_params = eltype.params()
        kdim_param = element_params['kdims']
        vdim_param = element_params['vdims']
        if util.is_dataframe(data):
            ndim = len(kdim_param.default) if kdim_param.default else None
            if kdims and vdims is None:
                vdims = [c for c in data.columns if c not in kdims]
            elif vdims and kdims is None:
                kdims = [c for c in data.columns if c not in vdims][:ndim]
            elif kdims is None and vdims is None:
                kdims = list(data.columns[:ndim])
                vdims = [] if ndim is None else list(data.columns[ndim:])
        else:
            # Check if data is of non-numeric type
            # Then use defined data type
            kdims = kdims if kdims else kdim_param.default
            vdims = vdims if vdims else vdim_param.default
            columns = [d.name if isinstance(d, Dimension) else d
                       for d in kdims+vdims]

            if ((isinstance(data, dict) and all(c in data for c in columns)) or
                (isinstance(data, NdElement) and all(c in data.dimensions() for c in columns))):
                data = cyODict(((d, data[d]) for d in columns))
            elif isinstance(data, dict) and not all(d in data for d in columns):
                column_data = zip(*((util.wrap_tuple(k)+util.wrap_tuple(v))
                                    for k, v in data.items()))
                data = cyODict(((c, col) for c, col in zip(columns, column_data)))
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    if eltype._1d:
                        data = np.atleast_2d(data).T
                    else:
                        data = (range(len(data)), data)
                else:
                    data = tuple(data[:, i] for i in range(data.shape[1]))

            if isinstance(data, tuple):
                data = [np.array(d) if not isinstance(d, np.ndarray) else d for d in data]
                if not cls.expanded(data):
                    raise ValueError('PandasInterface expects data to be of uniform shape.')
                data = pd.DataFrame.from_items([(c, d) for c, d in
                                                zip(columns, data)])
            else:
                data = pd.DataFrame(data, columns=columns)
        return data, {'kdims':kdims, 'vdims':vdims}, {}


    @classmethod
    def range(cls, columns, dimension):
        column = columns.data[columns.get_dimension(dimension).name]
        if column.dtype.kind == 'O':
            if (not isinstance(columns.data, pd.DataFrame) or
                        LooseVersion(pd.__version__) < '0.17.0'):
                column = column.sort(inplace=False)
            else:
                column = column.sort_values()
            return column.iloc[0], column.iloc[-1]
        else:
            return (column.min(), column.max())


    @classmethod
    def concat(cls, columns_objs):
        cast_objs = cls.cast(columns_objs)
        return pd.concat([col.data for col in cast_objs])


    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        index_dims = [columns.get_dimension(d) for d in dimensions]
        element_dims = [kdim for kdim in columns.kdims
                        if kdim not in index_dims]

        group_kwargs = {}
        if group_type != 'raw' and issubclass(group_type, Element):
            group_kwargs = dict(util.get_param_values(columns),
                                kdims=element_dims)
        group_kwargs.update(kwargs)

        data = [(k, group_type(v, **group_kwargs)) for k, v in
                columns.data.groupby(dimensions, sort=False)]
        if issubclass(container_type, NdMapping):
            with item_check(False):
                return container_type(data, kdims=index_dims)
        else:
            return container_type(data)


    @classmethod
    def aggregate(cls, columns, dimensions, function, **kwargs):
        data = columns.data
        cols = [d.name for d in columns.kdims if d in dimensions]
        vdims = columns.dimensions('value', True)
        reindexed = data.reindex(columns=cols+vdims)
        if len(dimensions):
            return reindexed.groupby(cols, sort=False).aggregate(function, **kwargs).reset_index()
        else:
            agg = reindexed.apply(function, **kwargs)
            return pd.DataFrame.from_items([(col, [v]) for col, v in
                                            zip(agg.index, agg.values)])


    @classmethod
    def unpack_scalar(cls, columns, data):
        """
        Given a columns object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if len(data) != 1 or len(data.columns) > 1:
            return data
        return data.iat[0,0]


    @classmethod
    def reindex(cls, columns, kdims=None, vdims=None):
        # DataFrame based tables don't need to be reindexed
        return columns.data


    @classmethod
    def sort(cls, columns, by=[]):
        import pandas as pd
        if not isinstance(by, list): by = [by]
        if not by: by = range(columns.ndims)
        cols = [columns.get_dimension(d).name for d in by]

        if (not isinstance(columns.data, pd.DataFrame) or
            LooseVersion(pd.__version__) < '0.17.0'):
            return columns.data.sort(columns=cols)
        return columns.data.sort_values(by=cols)


    @classmethod
    def select(cls, columns, selection_mask=None, **selection):
        df = columns.data
        if selection_mask is None:
            selection_mask = cls.select_mask(columns, selection)
        indexed = cls.indexed(columns, selection)
        df = df.ix[selection_mask]
        if indexed and len(df) == 1:
            return df[columns.vdims[0].name].iloc[0]
        return df


    @classmethod
    def values(cls, columns, dim, expanded=True, flat=True):
        data = columns.data[dim]
        if util.dd and isinstance(data, util.dd.Series):
            data = data.compute()
        if not expanded:
            return util.unique_array(data)
        return np.array(data)


    @classmethod
    def sample(cls, columns, samples=[]):
        data = columns.data
        mask = False
        for sample in samples:
            sample_mask = True
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                sample_mask = np.logical_and(sample_mask, data.iloc[:, i]==v)
            mask |= sample_mask
        return data[mask]


    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values, vdim):
        data = columns.data.copy()
        if dimension.name not in data:
            data.insert(dim_pos, dimension.name, values)
        return data


    @classmethod
    def dframe(cls, columns, dimensions):
        if dimensions:
            return columns.reindex(columns=dimensions)
        else:
            return columns.data.copy()


Interface.register(PandasInterface)
