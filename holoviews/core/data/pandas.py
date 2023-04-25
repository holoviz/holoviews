from collections import OrderedDict
from packaging.version import Version

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .interface import Interface, DataError
from ..dimension import dimension_name, Dimension
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .. import util
from .util import finite_range


class PandasAPI:
    """
    This class is used to describe the interface as having a pandas-like API.

    The reason to have this class is that it is not always
    possible to directly inherit from the PandasInterface.

    This class should not have any logic as it should be used like:
        if issubclass(interface, PandasAPI):
            ...
    """


class PandasInterface(Interface, PandasAPI):

    types = (pd.DataFrame,)

    datatype = 'dataframe'

    @classmethod
    def dimension_type(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
        idx = list(dataset.data.columns).index(name)
        return dataset.data.dtypes[idx].type

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        element_params = eltype.param.objects()
        kdim_param = element_params['kdims']
        vdim_param = element_params['vdims']
        if util.is_series(data):
            name = data.name or util.anonymous_dimension_label
            data = data.to_frame(name=name)
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

            if any(not isinstance(d, (str, Dimension)) for d in kdims+vdims):
                raise DataError(
                    "Having a non-string as a column name in a DataFrame is not supported."
                )

            # Handle reset of index if kdims reference index by name
            for kd in kdims:
                kd = dimension_name(kd)
                if kd in data.columns:
                    continue
                if any(kd == ('index' if name is None else name)
                       for name in index_names):
                    data = data.reset_index()
                    break

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
            columns = list(util.unique_iterator([dimension_name(d) for d in kdims+vdims]))

            if isinstance(data, dict) and all(c in data for c in columns):
                data = OrderedDict((d, data[d]) for d in columns)
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
                data = OrderedDict(((c, col) for c, col in zip(columns, column_data)))
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
                min_dims = (kdim_param.bounds[0] or 0) + (vdim_param.bounds[0] or 0)
                if any(d.ndim > 1 for d in data):
                    raise ValueError('PandasInterface cannot interpret multi-dimensional arrays.')
                elif len(data) < min_dims:
                    raise DataError('Data contains fewer columns than the %s element expects. Expected '
                                    'at least %d columns but found only %d columns.' %
                                    (eltype.__name__, min_dims, len(data)))
                elif not cls.expanded(data):
                    raise ValueError('PandasInterface expects data to be of uniform shape.')
                data = pd.DataFrame(dict(zip(columns, data)), columns=columns)
            elif ((isinstance(data, dict) and any(c not in data for c in columns)) or
                  (isinstance(data, list) and any(isinstance(d, dict) and c not in d for d in data for c in columns))):
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
        cols = list(dataset.data.columns)
        not_found = [d for d in dimensions if d not in cols]
        if not_found:
            raise DataError("Supplied data does not contain specified "
                            "dimensions, the following dimensions were "
                            "not found: %s" % repr(not_found), cls)


    @classmethod
    def range(cls, dataset, dimension):
        dimension = dataset.get_dimension(dimension, strict=True)
        column = dataset.data[dimension.name]
        if column.dtype.kind == 'O':
            if (not isinstance(dataset.data, pd.DataFrame) or
                util.pandas_version < Version('0.17.0')):
                column = column.sort(inplace=False)
            else:
                column = column.sort_values()
            try:
                column = column[~column.isin([None, pd.NA])]
            except Exception:
                pass
            if not len(column):
                return np.NaN, np.NaN
            return column.iloc[0], column.iloc[-1]
        else:
            if dimension.nodata is not None:
                column = cls.replace_value(column, dimension.nodata)
            cmin, cmax = finite_range(column, column.min(), column.max())
            if column.dtype.kind == 'M' and getattr(column.dtype, 'tz', None):
                return (cmin.to_pydatetime().replace(tzinfo=None),
                        cmax.to_pydatetime().replace(tzinfo=None))
            return cmin, cmax


    @classmethod
    def concat_fn(cls, dataframes, **kwargs):
        if util.pandas_version >= Version('0.23.0'):
            kwargs['sort'] = False
        return pd.concat(dataframes, **kwargs)


    @classmethod
    def concat(cls, datasets, dimensions, vdims):
        dataframes = []
        for key, ds in datasets:
            data = ds.data.copy()
            for d, k in zip(dimensions, key):
                data[d.name] = k
            dataframes.append(data)
        return cls.concat_fn(dataframes)


    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        index_dims = [dataset.get_dimension(d, strict=True) for d in dimensions]
        element_dims = [kdim for kdim in dataset.kdims
                        if kdim not in index_dims]

        group_kwargs = {}
        if group_type != 'raw' and issubclass(group_type, Element):
            group_kwargs = dict(util.get_param_values(dataset),
                                kdims=element_dims)
        group_kwargs.update(kwargs)

        # Propagate dataset
        group_kwargs['dataset'] = dataset.dataset

        group_by = [d.name for d in index_dims]
        if len(group_by) == 1 and util.pandas_version >= Version("1.5.0"):
            # Because of this deprecation warning from pandas 1.5.0:
            # In a future version of pandas, a length 1 tuple will be returned
            # when iterating over a groupby with a grouper equal to a list of length 1.
            # Don't supply a list with a single grouper to avoid this warning.
            group_by = group_by[0]
        data = [(k, group_type(v, **group_kwargs)) for k, v in
                dataset.data.groupby(group_by, sort=False)]
        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(data, kdims=index_dims)
        else:
            return container_type(data)


    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        data = dataset.data
        cols = [d.name for d in dataset.kdims if d in dimensions]
        vdims = dataset.dimensions('value', label='name')
        reindexed = data[cols+vdims]
        if function in [np.std, np.var]:
            # Fix for consistency with other backend
            # pandas uses ddof=1 for std and var
            fn = lambda x: function(x, ddof=0)
        else:
            fn = function
        if len(dimensions):
            # The reason to use `numeric_cols` is to prepare for when pandas will not
            # automatically drop columns that are not numerical for numerical
            # functions, e.g., `np.mean`.
            # pandas started warning about this in v1.5.0
            if fn in [np.size]:
                # np.size actually works with non-numerical columns
                numeric_cols = [
                    c for c in reindexed.columns if c not in cols
                ]
            else:
                numeric_cols = [
                    c for c, d in zip(reindexed.columns, reindexed.dtypes)
                    if is_numeric_dtype(d) and c not in cols
                ]
            grouped = reindexed.groupby(cols, sort=False)
            df = grouped[numeric_cols].aggregate(fn, **kwargs).reset_index()
        else:
            agg = reindexed.apply(fn, **kwargs)
            data = {col: [v] for col, v in zip(agg.index, agg.values)}
            df = pd.DataFrame(data, columns=list(agg.index))

        dropped = []
        for vd in vdims:
            if vd not in df.columns:
                dropped.append(vd)
        return df, dropped


    @classmethod
    def unpack_scalar(cls, dataset, data):
        """
        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if len(data) != 1 or len(data.columns) > 1:
            return data
        return data.iat[0,0]


    @classmethod
    def reindex(cls, dataset, kdims=None, vdims=None):
        # DataFrame based tables don't need to be reindexed
        return dataset.data


    @classmethod
    def mask(cls, dataset, mask, mask_value=np.nan):
        masked = dataset.data.copy()
        cols = [vd.name for vd in dataset.vdims]
        masked.loc[mask, cols] = mask_value
        return masked


    @classmethod
    def redim(cls, dataset, dimensions):
        column_renames = {k: v.name for k, v in dimensions.items()}
        return dataset.data.rename(columns=column_renames)


    @classmethod
    def sort(cls, dataset, by=[], reverse=False):
        cols = [dataset.get_dimension(d, strict=True).name for d in by]

        if (not isinstance(dataset.data, pd.DataFrame) or
            util.pandas_version < Version('0.17.0')):
            return dataset.data.sort(columns=cols, ascending=not reverse)
        return dataset.data.sort_values(by=cols, ascending=not reverse)


    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        df = dataset.data
        if selection_mask is None:
            selection_mask = cls.select_mask(dataset, selection)

        indexed = cls.indexed(dataset, selection)
        if isinstance(selection_mask, pd.Series):
            df = df[selection_mask]
        else:
            df = df.iloc[selection_mask]
        if indexed and len(df) == 1 and len(dataset.vdims) == 1:
            return df[dataset.vdims[0].name].iloc[0]
        return df


    @classmethod
    def values(
            cls,
            dataset,
            dim,
            expanded=True,
            flat=True,
            compute=True,
            keep_index=False,
    ):
        dim = dataset.get_dimension(dim, strict=True)
        data = dataset.data[dim.name]
        if keep_index:
            return data
        if data.dtype.kind == 'M' and getattr(data.dtype, 'tz', None):
            dts = [dt.replace(tzinfo=None) for dt in data.dt.to_pydatetime()]
            data = np.array(dts, dtype=data.dtype.base)
        if not expanded:
            return pd.unique(data)
        return data.values if hasattr(data, 'values') else data


    @classmethod
    def sample(cls, dataset, samples=[]):
        data = dataset.data
        mask = None
        for sample in samples:
            sample_mask = None
            if np.isscalar(sample): sample = [sample]
            for i, v in enumerate(sample):
                submask = data.iloc[:, i]==v
                if sample_mask is None:
                    sample_mask = submask
                else:
                    sample_mask &= submask
            if mask is None:
                mask = sample_mask
            else:
                mask |= sample_mask
        return data[mask]


    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        data = dataset.data.copy()
        if dimension.name not in data:
            data.insert(dim_pos, dimension.name, values)
        return data

    @classmethod
    def assign(cls, dataset, new_data):
        return dataset.data.assign(**new_data)

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
    def dframe(cls, dataset, dimensions):
        if dimensions:
            return dataset.data[dimensions]
        else:
            return dataset.data.copy()


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
