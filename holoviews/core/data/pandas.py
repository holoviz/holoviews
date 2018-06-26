from __future__ import absolute_import

from distutils.version import LooseVersion

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np
import pandas as pd

from .interface import Interface, DataError
from ..dimension import dimension_name
from ..element import Element
from ..dimension import OrderedDict as cyODict
from ..ndmapping import NdMapping, item_check
from .. import util


class PandasInterface(Interface):

    types = (pd.DataFrame if pd else None,)

    datatype = 'dataframe'

    @classmethod
    def dimension_type(cls, columns, dim):
        name = columns.get_dimension(dim, strict=True).name
        idx = list(columns.data.columns).index(name)
        return columns.data.dtypes[idx].type

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        element_params = eltype.params()
        kdim_param = element_params['kdims']
        vdim_param = element_params['vdims']
        if util.is_series(data):
            data = data.to_frame()
        if util.is_dataframe(data):
            ncols = len(data.columns)
            index_names = data.index.names if isinstance(data, pd.DataFrame) else [data.index.name]
            if index_names == [None]:
                index_names = ['index']
            if eltype._auto_indexable_1d and ncols == 1 and kdims is None:
                kdims = list(index_names)

            if isinstance(kdim_param.bounds[1], int):
                ndim = min([kdim_param.bounds[1], len(kdim_param.default)])
            else:
                ndim = None
            nvdim = vdim_param.bounds[1] if isinstance(vdim_param.bounds[1], int) else None
            if kdims and vdims is None:
                vdims = [c for c in data.columns if c not in kdims]
            elif vdims and kdims is None:
                kdims = [c for c in data.columns if c not in vdims][:ndim]
            elif kdims is None:
                kdims = list(data.columns[:ndim])
                if vdims is None:
                    vdims = [d for d in data.columns[ndim:((ndim+nvdim) if nvdim else None)]
                             if d not in kdims]
            elif kdims == [] and vdims is None:
                vdims = list(data.columns[:nvdim if nvdim else None])

            # Handle reset of index if kdims reference index by name
            for kd in kdims:
                kd = dimension_name(kd)
                if kd in data.columns:
                    continue
                if any(kd == ('index' if name is None else name)
                       for name in index_names):
                    data = data.reset_index()
                    break
            if any(isinstance(d, (np.int64, int)) for d in kdims+vdims):
                raise DataError("pandas DataFrame column names used as dimensions "
                                "must be strings not integers.", cls)

            if kdims:
                kdim = dimension_name(kdims[0])
                if eltype._auto_indexable_1d and ncols == 1 and kdim not in data.columns:
                    data = data.copy()
                    data.insert(0, kdim, np.arange(len(data)))

            for d in kdims+vdims:
                d = dimension_name(d)
                if len([c for c in data.columns if c == d]) > 1:
                    raise DataError('Dimensions may not reference duplicated DataFrame '
                                    'columns (found duplicate %r columns). If you want to plot '
                                    'a column against itself simply declare two dimensions '
                                    'with the same name. '% d, cls)
        else:
            # Check if data is of non-numeric type
            # Then use defined data type
            kdims = kdims if kdims else kdim_param.default
            vdims = vdims if vdims else vdim_param.default
            columns = [dimension_name(d) for d in kdims+vdims]

            if isinstance(data, dict) and all(c in data for c in columns):
                data = cyODict(((d, data[d]) for d in columns))
            elif isinstance(data, list) and len(data) == 0:
                data = {c: np.array([]) for c in columns}
            elif isinstance(data, (list, dict)) and data in ([], {}):
                data = None
            elif (isinstance(data, dict) and not all(d in data for d in columns) and
                  not any(isinstance(v, np.ndarray) for v in data.values())):
                column_data = sorted(data.items())
                k, v = column_data[0]
                if len(util.wrap_tuple(k)) != len(kdims) or len(util.wrap_tuple(v)) != len(vdims):
                    raise ValueError("Dictionary data not understood, should contain a column "
                                    "per dimension or a mapping between key and value dimension "
                                    "values.")
                column_data = zip(*((util.wrap_tuple(k)+util.wrap_tuple(v))
                                    for k, v in column_data))
                data = cyODict(((c, col) for c, col in zip(columns, column_data)))
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    if eltype._auto_indexable_1d and len(kdims)+len(vdims)>1:
                        data = (np.arange(len(data)), data)
                    else:
                        data = np.atleast_2d(data).T
                else:
                    data = tuple(data[:, i] for i in range(data.shape[1]))

            if isinstance(data, tuple):
                data = [np.array(d) if not isinstance(d, np.ndarray) else d for d in data]
                if not cls.expanded(data):
                    raise ValueError('PandasInterface expects data to be of uniform shape.')
                data = pd.DataFrame(dict(zip(columns, data)), columns=columns)
            elif isinstance(data, dict) and any(c not in data for c in columns):
                raise ValueError('PandasInterface could not find specified dimensions in the data.')
            else:
                data = pd.DataFrame(data, columns=columns)
        return data, {'kdims':kdims, 'vdims':vdims}, {}


    @classmethod
    def isscalar(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
        return len(dataset.data[name].unique()) == 1


    @classmethod
    def validate(cls, dataset, vdims=True):
        dim_types = 'all' if vdims else 'key'
        dimensions = dataset.dimensions(dim_types, label='name')
        not_found = [d for d in dimensions if d not in dataset.data.columns]
        if not_found:
            raise DataError("Supplied data does not contain specified "
                            "dimensions, the following dimensions were "
                            "not found: %s" % repr(not_found), cls)


    @classmethod
    def range(cls, columns, dimension):
        column = columns.data[columns.get_dimension(dimension, strict=True).name]
        if column.dtype.kind == 'O':
            if (not isinstance(columns.data, pd.DataFrame) or
                        LooseVersion(pd.__version__) < '0.17.0'):
                column = column.sort(inplace=False)
            else:
                column = column.sort_values()
            column = column[~column.isin([None])]
            if not len(column):
                return np.NaN, np.NaN
            return column.iloc[0], column.iloc[-1]
        else:
            return (column.min(), column.max())


    @classmethod
    def concat(cls, datasets, dimensions, vdims):
        dataframes = []
        for key, ds in datasets:
            data = ds.data.copy()
            for d, k in zip(dimensions, key):
                data[d.name] = k
            dataframes.append(data)
        return pd.concat(dataframes)


    @classmethod
    def groupby(cls, columns, dimensions, container_type, group_type, **kwargs):
        index_dims = [columns.get_dimension(d, strict=True) for d in dimensions]
        element_dims = [kdim for kdim in columns.kdims
                        if kdim not in index_dims]

        group_kwargs = {}
        if group_type != 'raw' and issubclass(group_type, Element):
            group_kwargs = dict(util.get_param_values(columns),
                                kdims=element_dims)
        group_kwargs.update(kwargs)

        group_by = [d.name for d in index_dims]
        data = [(k, group_type(v, **group_kwargs)) for k, v in
                columns.data.groupby(group_by, sort=False)]
        if issubclass(container_type, NdMapping):
            with item_check(False):
                return container_type(data, kdims=index_dims)
        else:
            return container_type(data)


    @classmethod
    def aggregate(cls, columns, dimensions, function, **kwargs):
        data = columns.data
        cols = [d.name for d in columns.kdims if d in dimensions]
        vdims = columns.dimensions('value', label='name')
        reindexed = data[cols+vdims]
        if function in [np.std, np.var]:
            # Fix for consistency with other backend
            # pandas uses ddof=1 for std and var
            fn = lambda x: function(x, ddof=0)
        else:
            fn = function
        if len(dimensions):
            grouped = reindexed.groupby(cols, sort=False)
            return grouped.aggregate(fn, **kwargs).reset_index()
        else:
            agg = reindexed.apply(fn, **kwargs)
            data = dict(((col, [v]) for col, v in zip(agg.index, agg.values)))
            return pd.DataFrame(data, columns=list(agg.index))


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
    def redim(cls, dataset, dimensions):
        column_renames = {k: v.name for k, v in dimensions.items()}
        return dataset.data.rename(columns=column_renames)


    @classmethod
    def sort(cls, columns, by=[], reverse=False):
        import pandas as pd
        cols = [columns.get_dimension(d, strict=True).name for d in by]

        if (not isinstance(columns.data, pd.DataFrame) or
            LooseVersion(pd.__version__) < '0.17.0'):
            return columns.data.sort(columns=cols, ascending=not reverse)
        return columns.data.sort_values(by=cols, ascending=not reverse)


    @classmethod
    def select(cls, columns, selection_mask=None, **selection):
        df = columns.data
        if selection_mask is None:
            selection_mask = cls.select_mask(columns, selection)
        indexed = cls.indexed(columns, selection)
        df = df.iloc[selection_mask]
        if indexed and len(df) == 1 and len(columns.vdims) == 1:
            return df[columns.vdims[0].name].iloc[0]
        return df


    @classmethod
    def values(cls, columns, dim, expanded=True, flat=True):
        dim = columns.get_dimension(dim, strict=True)
        data = columns.data[dim.name]
        if not expanded:
            return data.unique()
        return data.values


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
    def as_dframe(cls, dataset):
        """
        Returns the data of a Dataset as a dataframe avoiding copying
        if it already a dataframe type.
        """
        if issubclass(dataset.interface, PandasInterface):
            return dataset.data
        else:
            return dataset.dframe()


    @classmethod
    def dframe(cls, columns, dimensions):
        if dimensions:
            return columns.data[dimensions]
        else:
            return columns.data.copy()


    @classmethod
    def iloc(cls, dataset, index):
        rows, cols = index
        scalar = False
        columns = list(dataset.data.columns)
        if isinstance(cols, slice):
            cols = [d.name for d in dataset.dimensions()][cols]
        elif np.isscalar(cols):
            scalar = np.isscalar(rows)
            cols = [dataset.get_dimension(cols).name]
        else:
            cols = [dataset.get_dimension(d).name for d in index[1]]
        cols = [columns.index(c) for c in cols]
        if np.isscalar(rows):
            rows = [rows]

        if scalar:
            return dataset.data.iloc[rows[0], cols[0]]
        return dataset.data.iloc[rows, cols]


Interface.register(PandasInterface)
