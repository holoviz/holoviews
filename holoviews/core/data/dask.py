from __future__ import absolute_import

try:
    import itertools.izip as zip
except ImportError:
    pass

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.dataframe import DataFrame

from .. import util
from ..element import Element
from ..ndmapping import NdMapping, item_check, OrderedDict
from .interface import Interface
from .pandas import PandasInterface


class DaskInterface(PandasInterface):
    """
    The DaskInterface allows a Dataset objects to wrap a dask
    DataFrame object. Using dask allows loading data lazily
    and performing out-of-core operations on the data, making
    it possible to work on datasets larger than memory.

    The DaskInterface covers almost the complete API exposed
    by the PandasInterface with two notable exceptions:

    1) Sorting is not supported and any attempt at sorting will
       be ignored with an warning.
    2) Dask does not easily support adding a new column to an existing
       dataframe unless it is a scalar, add_dimension will therefore
       error when supplied a non-scalar value.
    4) Not all functions can be easily applied to a dask dataframe so
       some functions applied with aggregate and reduce will not work.
    """

    types = (DataFrame,)

    datatype = 'dask'

    default_partitions = 100

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        data, kdims, vdims = PandasInterface.init(eltype, data, kdims, vdims)
        if not isinstance(data, DataFrame):
            data = dd.from_pandas(data, npartitions=cls.default_partitions, sort=False)
        return data, kdims, vdims

    @classmethod
    def shape(cls, dataset):
        return (len(dataset.data), len(dataset.data.columns))

    @classmethod
    def range(cls, columns, dimension):
        column = columns.data[columns.get_dimension(dimension).name]
        if column.dtype.kind == 'O':
            column = np.sort(column[column.notnull()].compute())
            return column[0], column[-1]
        else:
            return dd.compute(column.min(), column.max())

    @classmethod
    def sort(cls, columns, by=[], reverse=False):
        columns.warning('Dask dataframes do not support sorting')
        return columns.data

    @classmethod
    def values(cls, columns, dim, expanded=True, flat=True):
        dim = columns.get_dimension(dim)
        data = columns.data[dim.name]
        if not expanded:
            data = data.unique()
        return data.compute().values

    @classmethod
    def select_mask(cls, dataset, selection):
        """
        Given a Dataset object and a dictionary with dimension keys and
        selection keys (i.e tuple ranges, slices, sets, lists or literals)
        return a boolean mask over the rows in the Dataset object that
        have been selected.
        """
        select_mask = None
        for dim, k in selection.items():
            if isinstance(k, tuple):
                k = slice(*k)
            masks = []
            alias = dataset.get_dimension(dim).name
            series = dataset.data[alias]
            if isinstance(k, slice):
                if k.start is not None:
                    # Workaround for dask issue #3392
                    kval = util.numpy_scalar_to_python(k.start)
                    masks.append(kval <= series)
                if k.stop is not None:
                    kval = util.numpy_scalar_to_python(k.stop)
                    masks.append(series < kval)
            elif isinstance(k, (set, list)):
                iter_slc = None
                for ik in k:
                    mask = series == ik
                    if iter_slc is None:
                        iter_slc = mask
                    else:
                        iter_slc |= mask
                masks.append(iter_slc)
            elif callable(k):
                masks.append(k(series))
            else:
                masks.append(series == k)
            for mask in masks:
                if select_mask is not None:
                    select_mask &= mask
                else:
                    select_mask = mask
        return select_mask

    @classmethod
    def select(cls, columns, selection_mask=None, **selection):
        df = columns.data
        if selection_mask is not None:
            return df[selection_mask]
        selection_mask = cls.select_mask(columns, selection)
        indexed = cls.indexed(columns, selection)
        df = df if selection_mask is None else df[selection_mask]
        if indexed and len(df) == 1 and len(columns.vdims) == 1:
            return df[columns.vdims[0].name].compute().iloc[0]
        return df

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

        data = []
        group_by = [d.name for d in index_dims]
        groupby = columns.data.groupby(group_by)
        if len(group_by) == 1:
            column = columns.data[group_by[0]]
            if column.dtype.name == 'category':
                try:
                    indices = ((ind,) for ind in column.cat.categories)
                except NotImplementedError:
                    indices = ((ind,) for ind in column.unique().compute())
            else:
                indices = ((ind,) for ind in column.unique().compute())
        else:
            group_tuples = columns.data[group_by].itertuples()
            indices = util.unique_iterator(ind[1:] for ind in group_tuples)
        for coord in indices:
            if any(isinstance(c, float) and np.isnan(c) for c in coord):
                continue
            if len(coord) == 1:
                coord = coord[0]
            group = group_type(groupby.get_group(coord), **group_kwargs)
            data.append((coord, group))
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
        dtypes = data.dtypes
        numeric = [c for c, dtype in zip(dtypes.index, dtypes.values)
                   if dtype.kind in 'iufc' and c in vdims]
        reindexed = data[cols+numeric]

        inbuilts = {'amin': 'min', 'amax': 'max', 'mean': 'mean',
                    'std': 'std', 'sum': 'sum', 'var': 'var'}
        if len(dimensions):
            groups = reindexed.groupby(cols)
            if (function.__name__ in inbuilts):
                agg = getattr(groups, inbuilts[function.__name__])()
            else:
                agg = groups.apply(function)
            return agg.reset_index()
        else:
            if (function.__name__ in inbuilts):
                agg = getattr(reindexed, inbuilts[function.__name__])()
            else:
                raise NotImplementedError
            return pd.DataFrame(agg.compute()).T

    @classmethod
    def unpack_scalar(cls, columns, data):
        """
        Given a columns object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if len(data.columns) > 1 or len(data) != 1:
            return data
        if isinstance(data, dd.DataFrame):
            data = data.compute()
        return data.iat[0,0]

    @classmethod
    def sample(cls, columns, samples=[]):
        data = columns.data
        dims = columns.dimensions('key', label='name')
        mask = None
        for sample in samples:
            if np.isscalar(sample): sample = [sample]
            for i, (c, v) in enumerate(zip(dims, sample)):
                dim_mask = data[c]==v
                if mask is None:
                    mask = dim_mask
                else:
                    mask |= dim_mask
        return data[mask]

    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values, vdim):
        data = columns.data
        if dimension.name not in data.columns:
            if not np.isscalar(values):
                err = ('Dask dataframe does not support assigning '
                       'non-scalar value.')
                raise NotImplementedError(err)
            data = data.assign(**{dimension.name: values})
        return data

    @classmethod
    def concat(cls, columns_objs):
        cast_objs = cls.cast(columns_objs)
        return dd.concat([col.data for col in cast_objs])

    @classmethod
    def dframe(cls, columns, dimensions):
        if dimensions:
            return columns.data[dimensions].compute()
        else:
            return columns.data.compute()

    @classmethod
    def nonzero(cls, dataset):
        return True

    @classmethod
    def iloc(cls, dataset, index):
        """
        Dask does not support iloc, therefore iloc will execute
        the call graph and lose the laziness of the operation.
        """
        rows, cols = index
        scalar = False
        if isinstance(cols, slice):
            cols = [d.name for d in dataset.dimensions()][cols]
        elif np.isscalar(cols):
            scalar = np.isscalar(rows)
            cols = [dataset.get_dimension(cols).name]
        else:
            cols = [dataset.get_dimension(d).name for d in index[1]]
        if np.isscalar(rows):
            rows = [rows]

        data = OrderedDict()
        for c in cols:
            data[c] = dataset.data[c].compute().iloc[rows].values
        if scalar:
            return data[cols[0]][0]
        return tuple(data.values())


Interface.register(DaskInterface)
