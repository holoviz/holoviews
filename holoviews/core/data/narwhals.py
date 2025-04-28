import builtins

import narwhals as nw
import numpy as np
from narwhals.dependencies import is_into_dataframe, is_into_series

from .. import util
from ..dimension import Dimension, dimension_name
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .interface import DataError, Interface
from .util import finite_range

_AGG_FUNC_LOOKUP = {
    builtins.sum: "sum",
    builtins.max: "max",
    builtins.min: "min",
    np.all: "all",
    np.any: "any",
    np.sum: "sum",
    np.nansum: "sum",
    np.mean: "mean",
    np.nanmean: "mean",
    # np.prod: "prod",
    # np.nanprod: "prod",
    np.std: "std",
    np.nanstd: "std",
    np.var: "var",
    np.nanvar: "var",
    np.median: "median",
    np.nanmedian: "median",
    np.max: "max",
    np.nanmax: "max",
    np.min: "min",
    np.nanmin: "min",
    np.size: "len",
}


class NarwhalsDtype:
    __slots__ = ("dtype",)

    def __init__(self, dtype):
        self.dtype = dtype

    @property
    def kind(self):
        if hasattr(self.dtype, "kind"):
            return self.dtype.kind
        return self._get_kind(self.dtype)

    @staticmethod
    def _get_kind(dtype: nw.dtypes.DType):
        if dtype.is_signed_integer():
            return "i"
        elif dtype.is_unsigned_integer():
            return "u"
        elif dtype.is_numeric():
            return "f"
        elif dtype.is_temporal():
            return "M"
        elif isinstance(dtype, nw.dtypes.Duration):
            return "m"
        elif isinstance(dtype, nw.dtypes.Boolean):
            return "b"
        elif isinstance(dtype, nw.dtypes.String):
            return "U"
        elif isinstance(dtype, nw.dtypes.Binary):
            return "S"
        else:
            return "O"

    def __repr__(self):
        return repr(self.dtype)


class NarwhalsInterface(Interface):
    datatype = "narwhals"

    @classmethod
    def applies(cls, obj):
        return is_into_dataframe(obj) or is_into_series(obj)

    @classmethod
    def dimension_type(cls, dataset, dim):
        return cls.dtype(dataset, dim)

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        element_params = eltype.param.objects()
        kdim_param = element_params["kdims"]
        vdim_param = element_params["vdims"]

        data = nw.from_native(data, allow_series=True)
        if isinstance(data, nw.Series):
            name = data.name or util.anonymous_dimension_label
            # Currently does not work: data = data.to_frame(name=name)
            data = data.rename(name).to_frame()
        if isinstance(data, (nw.DataFrame, nw.LazyFrame)):
            if isinstance(kdim_param.bounds[1], int):
                ndim = min([kdim_param.bounds[1], len(kdim_param.default)])
            else:
                ndim = None
            nvdim = (
                vdim_param.bounds[1] if isinstance(vdim_param.bounds[1], int) else None
            )
            columns = list(data.collect_schema())
            if kdims and vdims is None:
                vdims = [c for c in columns if c not in kdims]
            elif vdims and kdims is None:
                kdims = [c for c in columns if c not in vdims][:ndim]
            elif kdims is None:
                kdims = list(columns[:ndim])
                if vdims is None:
                    vdims = [
                        d
                        for d in columns[ndim : ((ndim + nvdim) if nvdim else None)]
                        if d not in kdims
                    ]
            elif kdims == [] and vdims is None:
                vdims = list(columns[: nvdim if nvdim else None])

            if any(not isinstance(d, (str, Dimension)) for d in kdims + vdims):
                raise DataError(
                    "Having a non-string as a column name in a DataFrame is not supported."
                )
            for d in kdims + vdims:
                d = dimension_name(d)
                if len([c for c in columns if c == d]) > 1:
                    raise DataError(
                        "Dimensions may not reference duplicated DataFrame "
                        f"columns (found duplicate {d!r} columns). If you want to plot "
                        "a column against itself simply declare two dimensions "
                        "with the same name.",
                        cls,
                    )
        return data, {"kdims": kdims, "vdims": vdims}, {}

    @classmethod
    def isscalar(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
        return len(dataset.data[name].unique()) == 1

    @classmethod
    def dtype(cls, dataset, dimension):
        dim = dataset.get_dimension(dimension, strict=True)
        nw_type = dataset.data.schema[dim.name]
        return NarwhalsDtype(nw_type)

    @classmethod
    def validate(cls, dataset, vdims=True):
        dim_types = "all" if vdims else "key"
        dimensions = dataset.dimensions(dim_types, label="name")
        cols = list(dataset.data.collect_schema())
        not_found = [d for d in dimensions if d not in cols]
        if not_found:
            raise DataError(
                "Supplied data does not contain specified "
                "dimensions, the following dimensions were "
                f"not found: {not_found!r}",
                cls,
            )

    @classmethod
    def range(cls, dataset, dimension):
        dimension = dataset.get_dimension(dimension, strict=True)
        column = dataset.data[dimension.name]
        if NarwhalsDtype(column.dtype).kind == "O":
            column = column.sort()
            if not len(column):
                return np.nan, np.nan
            return column.head(0), column.tail(0)
        else:
            if dimension.nodata is not None:
                column = column.fill_null(dimension.nodata)
            cmin, cmax = finite_range(column, column.min(), column.max())
            return cmin, cmax

    @classmethod
    def concat_fn(cls, dataframes, **kwargs):
        return nw.concat(dataframes, **kwargs)

    @classmethod
    def concat(cls, datasets, dimensions, vdims):
        dataframes = []
        for key, ds in datasets:
            data = ds.data.clone()
            new_columns = [
                nw.lit(val).alias(dim.name)
                for dim, val in zip(dimensions, key, strict=None)
            ]
            data = data.with_columns(new_columns)
            dataframes.append(data)
        return cls.concat_fn(dataframes)

    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        index_dims = [dataset.get_dimension(d, strict=True) for d in dimensions]
        element_dims = [kdim for kdim in dataset.kdims if kdim not in index_dims]

        group_kwargs = {}
        if group_type != "raw" and issubclass(group_type, Element):
            group_kwargs = dict(util.get_param_values(dataset), kdims=element_dims)
        group_kwargs.update(kwargs)
        group_kwargs["dataset"] = dataset.dataset

        group_by = [d.name for d in index_dims]
        data = [
            (k, group_type(v, **group_kwargs))
            for k, v in dataset.data.group_by(group_by)
        ]

        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(data, kdims=index_dims)
        else:
            return container_type(data)

    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        cols = [d.name for d in dataset.kdims if d in dimensions]
        vdims = dataset.dimensions("value", label="name")
        reindexed = cls.dframe(dataset, dimensions=cols + vdims)
        expr = getattr(nw.col("*"), _AGG_FUNC_LOOKUP.get(function, function))()
        if len(dimensions):
            columns = reindexed.collect_schema()
            if function in [np.size]:
                numeric_cols = [c for c in columns if c not in cols]
            else:
                numeric_cols = [
                    k
                    for k, v in columns.items()
                    if isinstance(v, nw.dtypes.NumericType)
                ]
            grouped = reindexed.select(numeric_cols + cols).groupby(cols)
            df = grouped.agg(expr, **kwargs)
        else:
            df = reindexed.select(expr, **kwargs)

        dropped = []
        columns = list(df.collect_schema())
        for vd in vdims:
            if vd not in columns:
                dropped.append(vd)
        return df, dropped

    @classmethod
    def unpack_scalar(cls, dataset, data):
        """Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.

        """
        cols = data.collect_schema()
        if len(cols) > 1:
            return data
        is_lazy = isinstance(data, nw.LazyFrame)
        size = data.select(nw.col(cols[0]).len())
        size = size.collect() if is_lazy else size
        if size != 1:
            return data
        return (data.collect() if is_lazy else data).item()

    @classmethod
    def mask(cls, dataset, mask, mask_value=None):
        data = dataset.data
        if not dataset.vdims:
            return data

        if data.implementation != nw.Implementation.POLARS:
            # polars will convert None to Null
            mask_value = np.nan

        cols = [vd.name for vd in dataset.vdims]
        mask_series = nw.new_series(
            name="__mask__", values=mask, backend=data.implementation
        )
        return (
            data.with_columns(mask_series)
            .with_columns(
                [
                    nw.when(nw.col("__mask__"))
                    .then(nw.col(col))
                    .otherwise(mask_value)
                    .alias(col)
                    for col in cols
                ]
            )
            .drop("__mask__")
        )

    @classmethod
    def redim(cls, dataset, dimensions):
        column_renames = {k: v.name for k, v in dimensions.items()}
        return dataset.data.rename(column_renames)

    @classmethod
    def sort(cls, dataset, by=None, reverse=False):
        if by is None:
            by = []
        cols = [dataset.get_dimension(d, strict=True).name for d in by]
        return dataset.data.sort_values(by=cols, descending=reverse)

    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        df = dataset.data
        if selection_mask is None:
            column_sel = {k: v for k, v in selection.items()}
            if column_sel:
                selection_mask = cls.select_mask(dataset, column_sel)

        if selection_mask is not None:
            df = df.filter(selection_mask)
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
        if getattr(data.dtype, "time_zone", False):
            data = data.dt.replace_time_zone(None)
        if not expanded:
            data = data.unique()
        if isinstance(data, nw.LazyFrame):
            data = data.collect()
        return data

    @classmethod
    def sample(cls, dataset, samples=None):
        if samples is None:
            samples = []
        data = dataset.data
        columns = list(data.collect_schema())
        mask = None
        for sample in samples:
            sample_mask = None
            if np.isscalar(sample):
                sample = [sample]
            for col, value in zip(columns, sample, strict=False):
                submask = nw.col(col) == value
                if sample_mask is None:
                    sample_mask = submask
                else:
                    sample_mask &= submask
            if mask is None:
                mask = sample_mask
            else:
                mask |= sample_mask
        return data.filter(mask)

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        data = dataset.data.clone()
        if dimension.name not in data:
            cols = list(data.collect_schema())
            cols = [cols[:dim_pos], dimension.name, cols[dim_pos:]]
            data = data.with_columns(**{dimension.name: values})[cols]
        return data

    @classmethod
    def assign(cls, dataset, new_data):
        return dataset.data.with_columns(**new_data)

    @classmethod
    def as_dframe(cls, dataset):
        """Returns the data of a Dataset as a dataframe avoiding copying
        if it already a dataframe type.

        """
        if issubclass(dataset.interface, NarwhalsInterface):
            return dataset.data
        else:
            return dataset.dframe()

    @classmethod
    def dframe(cls, dataset, dimensions):
        if dimensions:
            return dataset.data.select(dimensions).clone()
        else:
            return dataset.data.clone()

    @classmethod
    def iloc(cls, dataset, index):
        rows, cols = index
        scalar = False
        if isinstance(cols, slice):
            cols = [d.name for d in dataset.dimensions()][cols]
        elif np.isscalar(cols):
            scalar = np.isscalar(rows)
            dim = dataset.get_dimension(cols)
            if dim is None:
                raise ValueError("column is out of bounds")
            cols = [dim.name]
        else:
            cols = [dataset.get_dimension(d).name for d in cols]
        if np.isscalar(rows):
            rows = [rows]

        data = dataset.data
        indexes = cls.indexes(data)
        columns = list(data.collect_schema())
        id_cols = [columns.index(c) for c in cols if c not in indexes]
        if not id_cols:
            if len(indexes) > 1:
                data = data.index.to_frame()[cols].iloc[rows].reset_index(drop=True)
                data = data.values.ravel()[0] if scalar else data
            else:
                data = data.index.values[rows[0]] if scalar else data.index[rows]
            return data
        if scalar:
            return data.iloc[rows[0], id_cols[0]]
        return data.iloc[rows, id_cols]

    @classmethod
    def nonzero(cls, dataset):
        if isinstance(dataset.data, nw.LazyFrame):
            return True
        else:
            return super().nonzero(dataset)


Interface.register(NarwhalsInterface)
