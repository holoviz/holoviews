import sys
import typing

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
    def init(cls, eltype, data, kdims, vdims):
        element_params = eltype.param.objects()
        kdim_param = element_params["kdims"]
        vdim_param = element_params["vdims"]

        ncols = len(data.columns)

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
                vdims = [
                    d
                    for d in data.columns[ndim : ((ndim + nvdim) if nvdim else None)]
                    if d not in kdims
                ]
        elif kdims == [] and vdims is None:
            vdims = list(data.columns[: nvdim if nvdim else None])
        return data, {"kdims": kdims, "vdims": vdims}, {}

    @classmethod
    def length(self, dataset):
        return getattr(dataset, "length", dataset.data[[]].count().execute())

    @classmethod
    def nonzero(cls, dataset):
        return bool(len(dataset.data[[]].head(1).execute()))

    @classmethod
    def range(cls, dataset, dimension):
        column = dataset.data[dataset.get_dimension(dimension, strict=True).name]
        return tuple(
            dataset.data.aggregate([column.min(), column.max()]).execute().values[0, :]
        )

    @classmethod
    def values(
        cls, dataset, dim, expanded=True, flat=True, compute=True, keep_index=False
    ):
        dim = dataset.get_dimension(dim, strict=True)
        data = dataset.data[dim.name]
        if not expanded:
            return data.distinct().execute()
        return data if keep_index else data.execute()

        data = dataset.data[dataset.get_dimension(dim, strict=True).name]
        if not expanded:
            data = data.distinct()
        return data.execute() if keep_index else data.execute().values

    @classmethod
    def shape(cls, dataset) -> typing.Tuple[int, int]:
        return cls.length(dataset), len(dataset.data)

    @classmethod
    def array(cls, dataset, dimensions):
        return dataset.data[
            [dataset.get_dimension_index(d) for d in dimensions]
            if dimensions
            else slice(None)
        ]

    @classmethod
    def dtype(cls, dataset, dimension):
        return dataset.data.head(0).execute().dtypes[dimension.name]

    @classmethod
    def sort(cls, dataset, by=[], reverse=False):
        return dataset.data.sort_by([(x, reverse) for x in by])

    @classmethod
    def redim(cls, dataset, dimensions):
        return dataset.data.mutate(**{k: v.name for k, v in dimensions.items()})

    validate = pandas.PandasInterface.validate
    reindex = pandas.PandasInterface.reindex

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


Interface.register(IbisInterface)
