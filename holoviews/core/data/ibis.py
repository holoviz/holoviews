import sys
import numpy

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from .. import util
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .interface import Interface
from . import pandas
from .util import cached


class IbisInterface(Interface):

    types = ()

    datatype = "ibis"

    default_partitions = 100

    zero_indexed_backend_modules = [
        'ibis.backends.omniscidb.client',
    ]

    # the rowid is needed until ibis updates versions
    @classmethod
    def has_rowid(cls):
        import ibis.expr.operations
        return hasattr(ibis.expr.operations, "RowID")

    @classmethod
    def is_rowid_zero_indexed(cls, data):
        from ibis.client import find_backends, validate_backends
        (backend,) = validate_backends(list(find_backends(data)))
        return type(backend).__module__ in cls.zero_indexed_backend_modules

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
                    key
                    for key in data.columns[ndim : ((ndim + nvdim) if nvdim else None)]
                    if key not in keys
                ]
        elif keys == [] and values is None:
            values = list(data.columns[: nvdim if nvdim else None])
        return data, dict(kdims=keys, vdims=values), {}

    @classmethod
    def compute(cls, dataset):
        return dataset.clone(dataset.data.execute())

    @classmethod
    def persist(cls, dataset):
        return cls.compute(dataset)

    @classmethod
    @cached
    def length(self, dataset):
        # Get the length by counting the length of an empty query.
        return dataset.data[[]].count().execute()

    @classmethod
    @cached
    def nonzero(cls, dataset):
        # Make an empty query to see if a row is returned.
        return bool(len(dataset.data[[]].head(1).execute()))

    @classmethod
    @cached
    def range(cls, dataset, dimension):
        dimension = dataset.get_dimension(dimension, strict=True)
        if cls.dtype(dataset, dimension).kind in 'SUO':
            return None, None
        if dimension.nodata is not None:
            return Interface.range(dataset, dimension)
        column = dataset.data[dimension.name]
        return tuple(
            dataset.data.aggregate([column.min(), column.max()]).execute().values[0, :]
        )

    @classmethod
    @cached
    def values(
        cls,
        dataset,
        dimension,
        expanded=True,
        flat=True,
        compute=True,
        keep_index=False,
    ):
        dimension = dataset.get_dimension(dimension, strict=True)
        data = dataset.data[dimension.name]
        if not expanded:
            data = data.distinct()
        return data if keep_index or not compute else data.execute().values

    @classmethod
    def histogram(cls, expr, bins, density=True, weights=None):
        bins = numpy.asarray(bins)
        bins = [int(v) if bins.dtype.kind in 'iu' else float(v) for v in bins]
        binned = expr.bucket(bins).name('bucket')
        hist = numpy.zeros(len(bins)-1)
        hist_bins = binned.value_counts().sort_by('bucket').execute()
        for b, v in zip(hist_bins['bucket'], hist_bins['count']):
            if numpy.isnan(b):
                continue
            hist[int(b)] = v
        if weights is not None:
            raise NotImplementedError("Weighted histograms currently "
                                      "not implemented for IbisInterface.")
        if density:
            hist = hist/expr.count().execute()
        return hist, bins

    @classmethod
    @cached
    def shape(cls, dataset):
        return cls.length(dataset), len(dataset.data.columns)

    @classmethod
    @cached
    def dtype(cls, dataset, dimension):
        dimension = dataset.get_dimension(dimension)
        return dataset.data.head(0).execute().dtypes[dimension.name]

    dimension_type = dtype

    @classmethod
    def sort(cls, dataset, by=[], reverse=False):
        return dataset.data.sort_by([(dataset.get_dimension(x).name, not reverse) for x in by])

    @classmethod
    def redim(cls, dataset, dimensions):
        return dataset.data.mutate(
            **{v.name: dataset.data[k] for k, v in dimensions.items()}
        )

    validate = pandas.PandasInterface.validate
    reindex = pandas.PandasInterface.reindex

    @classmethod
    def _index_ibis_table(cls, data):
        import ibis
        if not cls.has_rowid():
            raise ValueError(
                "iloc expressions are not supported for ibis version %s."
                % ibis.__version__
            )

        if "hv_row_id__" in data.columns:
            return data
        if cls.is_rowid_zero_indexed(data):
            return data.mutate(hv_row_id__=data.rowid())
        else:
            return data.mutate(hv_row_id__=data.rowid() - 1)

    @classmethod
    def iloc(cls, dataset, index):
        rows, columns = index
        scalar = all(map(util.isscalar, index))

        if isinstance(columns, slice):
            columns = [x.name for x in dataset.dimensions()[columns]]
        elif numpy.isscalar(columns):
            columns = [dataset.get_dimension(columns).name]
        else:
            columns = [dataset.get_dimension(d).name for d in columns]

        data = cls._index_ibis_table(dataset.data[columns])

        if scalar:
            return (
                data.filter(data.hv_row_id__ == rows)[columns]
                .head(1)
                .execute()
                .iloc[0, 0]
            )

        if isinstance(rows, slice):
            # We should use a pseudo column for the row number but i think that is still awaiting
            # a pr on ibis
            if any(x is not None for x in (rows.start, rows.stop, rows.step)):
                predicates = []
                if rows.start:
                    predicates += [data.hv_row_id__ >= rows.start]
                if rows.stop:
                    predicates += [data.hv_row_id__ < rows.stop]

                return data.filter(predicates).drop(["hv_row_id__"])
        else:
            if not isinstance(rows, Iterable):
                rows = [rows]
            return data.filter([data.hv_row_id__.isin(rows)]).drop(["hv_row_id__"])
        return data.drop(["hv_row_id__"])

    @classmethod
    def unpack_scalar(cls, dataset, data):
        """
        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if len(data.columns) > 1 or data[[]].count().execute() != 1:
            return data
        return data.execute().iat[0, 0]

    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        # aggregate the necesary dimensions
        index_dims = [dataset.get_dimension(d, strict=True) for d in dimensions]
        element_dims = [kdim for kdim in dataset.kdims if kdim not in index_dims]

        group_kwargs = {}
        if group_type != "raw" and issubclass(group_type, Element):
            group_kwargs = dict(util.get_param_values(dataset), kdims=element_dims)
        group_kwargs.update(kwargs)
        group_kwargs["dataset"] = dataset.dataset

        group_by = [d.name for d in index_dims]

        # execute a query against the table to find the unique groups.
        groups = dataset.data.groupby(group_by).aggregate().execute()

        # filter each group based on the predicate defined.
        data = [
            (
                tuple(s.values.tolist()),
                group_type(
                    dataset.data.filter(
                        [dataset.data[k] == v for k, v in s.to_dict().items()]
                    ),
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
    def assign(cls, dataset, new_data):
        return dataset.data.mutate(**new_data)

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        import ibis
        data = dataset.data
        if dimension.name not in data.columns:
            if not isinstance(values, ibis.Expr) and not numpy.isscalar(values):
                raise ValueError("Cannot assign %s type as a Ibis table column, "
                                 "expecting either ibis.Expr or scalar."
                                 % type(values).__name__)
            data = data.mutate(**{dimension.name: values})
        return data

    @classmethod
    @cached
    def isscalar(cls, dataset, dim):
        return (
            dataset.data[dataset.get_dimension(dim, strict=True).name]
            .distinct()
            .count()
            .compute()
            == 1
        )

    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        if selection_mask is None:
            selection_mask = cls.select_mask(dataset, selection)
        indexed = cls.indexed(dataset, selection)
        data = dataset.data

        if isinstance(selection_mask, numpy.ndarray):
            data = cls._index_ibis_table(data)
            if selection_mask.dtype == numpy.dtype("bool"):
                selection_mask = numpy.where(selection_mask)[0]
            data = data.filter(
                data["hv_row_id__"].isin(list(map(int, selection_mask)))
            ).drop(["hv_row_id__"])
        elif selection_mask is not None and not (isinstance(selection_mask, list) and not selection_mask):
            data = data.filter(selection_mask)

        if indexed and data.count().execute() == 1 and len(dataset.vdims) == 1:
            return data[dataset.vdims[0].name].execute().iloc[0]
        return data

    @classmethod
    def select_mask(cls, dataset, selection):
        import ibis
        predicates = []
        for dim, object in selection.items():
            if isinstance(object, tuple):
                object = slice(*object)
            alias = dataset.get_dimension(dim).name
            column = dataset.data[alias]
            if isinstance(object, slice):
                if object.start is not None:
                    # Workaround for dask issue #3392
                    bound = util.numpy_scalar_to_python(object.start)
                    predicates.append(bound <= column)
                if object.stop is not None:
                    bound = util.numpy_scalar_to_python(object.stop)
                    predicates.append(column < bound)
            elif isinstance(object, (set, list)):
                # rowid conditions
                condition = None
                for id in object:
                    predicate = column == id
                    condition = (
                        predicate if condition is None else condition | predicate
                    )
                if condition is not None:
                    predicates.append(condition)
            elif callable(object):
                predicates.append(object(column))
            elif isinstance(object, ibis.Expr):
                predicates.append(object)
            else:
                predicates.append(column == object)
        return predicates

    @classmethod
    def sample(cls, dataset, samples=[]):
        import ibis
        dims = dataset.dimensions()
        data = dataset.data
        if all(util.isscalar(s) or len(s) == 1 for s in samples):
            items = [s[0] if isinstance(s, tuple) else s for s in samples]
            return data[data[dims[0].name].isin(items)]

        predicates = None
        for sample in samples:
            if util.isscalar(sample):
                sample = [sample]
            if not sample:
                continue
            predicate = None
            for i, v in enumerate(sample):
                p = data[dims[i].name] == ibis.literal(util.numpy_scalar_to_python(v))
                if predicate is None:
                    predicate = p
                else:
                    predicate &= p
            if predicates is None:
                predicates = predicate
            else:
                predicates |= predicate
        return data if predicates is None else data.filter(predicates)

    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        import ibis.expr.operations
        data = dataset.data
        columns = [d.name for d in dataset.kdims if d in dimensions]
        values = dataset.dimensions("value", label="name")
        new = data[columns + values]

        function = {
            numpy.min: ibis.expr.operations.Min,
            numpy.nanmin: ibis.expr.operations.Min,
            numpy.max: ibis.expr.operations.Max,
            numpy.nanmax: ibis.expr.operations.Max,
            numpy.mean: ibis.expr.operations.Mean,
            numpy.nanmean: ibis.expr.operations.Mean,
            numpy.std: ibis.expr.operations.StandardDev,
            numpy.nanstd: ibis.expr.operations.StandardDev,
            numpy.sum: ibis.expr.operations.Sum,
            numpy.nansum: ibis.expr.operations.Sum,
            numpy.var: ibis.expr.operations.Variance,
            numpy.nanvar: ibis.expr.operations.Variance,
            len: ibis.expr.operations.Count,
        }.get(function, function)

        if len(dimensions):
            selection = new.groupby(columns)
            if function is numpy.count_nonzero:
                aggregation = selection.aggregate(
                    **{
                        x: ibis.expr.operations.Count(new[x], where=new[x] != 0).to_expr()
                        for x in new.columns
                        if x not in columns
                    }
                )
            else:
                aggregation = selection.aggregate(
                    **{
                        x: function(new[x]).to_expr()
                        for x in new.columns
                        if x not in columns
                    }
                )
        else:
            aggregation = new.aggregate(
                **{x: function(new[x]).to_expr() for x in new.columns}
            )

        dropped = [x for x in values if x not in data.columns]
        return aggregation, dropped

    @classmethod
    @cached
    def mask(cls, dataset, mask, mask_value=numpy.nan):
        raise NotImplementedError('Mask is not implemented for IbisInterface.')

    @classmethod
    @cached
    def dframe(cls, dataset, dimensions):
        return dataset.data[dimensions].execute()

Interface.register(IbisInterface)
