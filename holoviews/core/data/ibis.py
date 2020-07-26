import sys
import typing

import holoviews

from .interface import Interface


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
    def shape(cls, dataset: holoviews.Dataset) -> typing.Tuple[int, int]:
        return cls.length(dataset), len(dataset.data)

    @classmethod
    def array(cls, dataset: holoviews.Dataset, dimensions: holoviews.Dimension):
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

    validate = holoviews.core.data.pandas.PandasInterface.validate
    reindex = holoviews.core.data.pandas.PandasInterface.reindex


Interface.register(IbisInterface)
