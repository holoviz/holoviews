import sys
import typing
import numpy
import holoviews

from .interface import Interface
from . import pandas


class IbisInterface(Interface):

    types = ()

    datatype = "ibis"

    default_partitions = 100

    @classmethod
    def loaded(cls):
        return "ibis" in sys.modules

    @classmethod
    def applies(cls, obj):
        if not cls.loaded():
            return False
        from ibis.expr.types import Expr

        return isinstance(obj, Expr)

    @classmethod
    def init(cls, eltype, data, keys, values):
        params = eltype.param.objects()
        index = params["kdims"]
        columns = params["vdims"]

        if isinstance(index.bounds[1], int):
            ndim = min([index.bounds[1], len(index.default)])
        else:
            ndim = None
        nvdim = columns.bounds[1] if isinstance(columns.bounds[1], int) else None
        if keys and values is None:
            values = [c for c in data.columns if c not in keys]
        elif values and keys is None:
            keys = [c for c in data.columns if c not in values][:ndim]
        elif keys is None:
            keys = list(data.columns[:ndim])
            if values is None:
                values = [
                    d
                    for key in data.columns[ndim : ((ndim + nvdim) if nvdim else None)]
                    if key not in keys
                ]
        elif keys == [] and values is None:
            values = list(data.columns[: nvdim if nvdim else None])
        return data, dict(kdims=keys, vdims=values), {}

    @classmethod
    def length(self, dataset):
        # Get the length by counting the length of an empty query.
        return getattr(dataset, "length", dataset.data[[]].count().execute())

    @classmethod
    def nonzero(cls, dataset):
        # Make an empty query to see if a row is returned.
        return bool(len(dataset.data[[]].head(1).execute()))

    @classmethod
    def range(cls, dataset, dimension):
        column = dataset.data[dataset.get_dimension(dimension, strict=True).name]
        return tuple(
            dataset.data.aggregate([column.min(), column.max()]).execute().values[0, :]
        )

    @classmethod
    def values(
        cls,
        dataset,
        dimension,
        expanded=True,
        flat=True,
        compute=True,
        keep_index=False,
    ) -> numpy.ndarray:
        dimension = dataset.get_dimension(dimension, strict=True)
        data = dataset.data[dimension.name]
        if not expanded:
            return data.distinct().execute().values
        return data if keep_index else data.execute().values

        data = dataset.data[dataset.get_dimension(dimension, strict=True).name]
        if not expanded:
            data = data.distinct()
        return data.execute().values if keep_index else data.execute().values

    @classmethod
    def shape(cls, dataset) -> typing.Tuple[int, int]:
        return cls.length(dataset), len(dataset.data.columns)

    @classmethod
    def dtype(cls, dataset, dimension):
        return dataset.data.head(0).execute().dtypes[dimension.name]

    @classmethod
    def sort(cls, dataset, by=[], reverse=False):
        return dataset.data.sort_by([(x, reverse) for x in by])

    @classmethod
    def redim(cls, dataset, dimensions: dict):
        return dataset.data.mutate(
            **{k: dataset.data[v.name] for k, v in dimensions.items()}
        )

    validate = pandas.PandasInterface.validate
    reindex = pandas.PandasInterface.reindex

    @classmethod
    def iloc(cls, dataset, index):
        rows, columns = index
        scalar = False
        data_columns = list(dataset.data.columns)
        if isinstance(columns, slice):
            columns = [d.name for d in dataset.dimensions()][columns]
        elif np.isscalar(cols):
            columns = [dataset.get_dimension(columns).name]
        else:
            columns: typing.List[str] = [
                dataset.get_dimension(d).name for d in index[1]
            ]
        columns = [data_columns.index(c) for c in columns]

        if scalar:
            data = dataset.data[[columns[0]]].mutate(hv_row_id__=ibis.row_number())
            return dataset.data[[columns[0]]].filter([data.hv_row_id__ == rows[0]])

        data = dataset.data[columns]

        if isinstance(rows, slice):
            if any(x is not None for x in (rows.start, rows.stop, rows.step)):
                predicates = []
                data = cls.assign(dataset, dict(hv_row_id__=ibis.row_number()))

                if rows.start:
                    predicates += [data.hv_row_id__ > ibis.literal(rows.start)]
                if rows.stop:
                    predicates += [data.hv_row_id__ < ibis.literal(rows.stop)]

                return data.filter(predicates).drop(["hv_row_id__"])
        return data

    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        # aggregate the necesary dimensions
        index_dims: typing.List[str] = [
            dataset.get_dimension(d, strict=True) for d in dimensions
        ]
        element_dims: typing.List[str] = [
            kdim for kdim in dataset.kdims if kdim not in index_dims
        ]

        # some metdata shit
        group_kwargs: dict = {}
        if group_type != "raw" and issubclass(group_type, Element):
            group_kwargs = dict(util.get_param_values(dataset), kdims=element_dims)
        group_kwargs.update(kwargs)
        group_kwargs["dataset"] = dataset.dataset

        group_by: typing.List[str] = [d.name for d in index_dims]

        # execute a query against the table to find the unique groups.
        groups: "pd.DataFrame" = dataset.data.groupby(group_by).aggregate().execute()

        # filter each group based on the predicate defined.
        data: typing.List[typing.Tuple[str, group_type]] = [
            (
                tuple(s.value.tolist()),
                group_type(
                    v.filter([v[k] == v for k, v in s.to_dict().items()]),
                    **group_kwargs
                ),
            )
            for i, s in groups.iterrows()
        ]
        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(data, kdims=index_dims)
        else:
            return container_type(data)

    @classmethod
    def assign(cls, dataset, new_data: typing.Dict[str, "ibis.Expr"]):
        return dataset.data.mutate(**new_data)

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        data = dataset.data
        if dimension.name not in data.columns:
            if not numpy.isscalar(values):
                err = "ibis dataframe does not support assigning " "non-scalar value."
                raise NotImplementedError(err)
            data = data.mutate(**{dimension.name: values})
        return data

    @classmethod
    def isscalar(cls, dataset, dim):
        return (
            dataset.data[dataset.get_dimension(dim, strict=True).name]
            .distinct()
            .count()
            .compute()
            == 1
        )


Interface.register(IbisInterface)
