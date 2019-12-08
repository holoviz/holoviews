from __future__ import absolute_import

import sys
import warnings

try:
    import itertools.izip as zip
except ImportError:
    pass

from itertools import product

import numpy as np

from .. import util
from ..dimension import dimension_name
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .interface import DataError, Interface
from .pandas import PandasInterface


class cuDFInterface(PandasInterface):
    """
    The cuDFInterface allows a Dataset objects to wrap a cuDF
    DataFrame object. Using cuDF allows working with columnar
    data on a GPU. Most operation leave the data in GPU memory,
    however to plot the data it has to be loaded into memory.

    The cuDFInterface covers almost the complete API exposed
    by the PandasInterface with two notable exceptions:

    1) Sorting is not supported and any attempt at sorting will
       be ignored with an warning.
    2) cuDF does not easily support adding a new column to an existing
       dataframe unless it is a scalar, add_dimension will therefore
       error when supplied a non-scalar value.
    3) Not all functions can be easily applied to a dask dataframe so
       some functions applied with aggregate and reduce will not work.
    """

    datatype = 'cuDF'

    types = ()

    @classmethod
    def loaded(cls):
        return 'cudf' in sys.modules

    @classmethod
    def applies(cls, obj):
        if not cls.loaded():
            return False
        import cudf
        return isinstance(obj, (cudf.DataFrame, cudf.Series))

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        import cudf

        element_params = eltype.param.objects()
        kdim_param = element_params['kdims']
        vdim_param = element_params['vdims']
        ncols = len(data.columns)

        if isinstance(data, cudf.Series):
            data = data.to_frame()

        index_names = [data.index.name]
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
            raise DataError("cudf DataFrame column names used as dimensions "
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
        return data, {'kdims':kdims, 'vdims':vdims}, {}

    

    @classmethod
    def range(cls, dataset, dimension):
        column = dataset.data[dataset.get_dimension(dimension, strict=True).name]
        if column.dtype.kind == 'O':
            return np.NaN, np.NaN
        else:
            return (column.min(), column.max())


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True, compute=True,
               keep_index=False):
        dim = dataset.get_dimension(dim, strict=True)
        data = dataset.data[dim.name]
        if not expanded:
            data = data.unique()
            return data.to_array() if compute else data
        elif keep_index:
            return data
        elif compute:
            return data.to_array()
        return data


    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        # Get dimensions information
        dimensions = [dataset.get_dimension(d).name for d in dimensions]
        kdims = [kdim for kdim in dataset.kdims if kdim not in dimensions]

        # Update the kwargs appropriately for Element group types
        group_kwargs = {}
        group_type = dict if group_type == 'raw' else group_type
        if issubclass(group_type, Element):
            group_kwargs.update(util.get_param_values(dataset))
            group_kwargs['kdims'] = kdims
        group_kwargs.update(kwargs)

        # Find all the keys along supplied dimensions
        keys = product(*(dataset.data[dimensions[0]].unique() for d in dimensions))

        # Iterate over the unique entries applying selection masks
        grouped_data = []
        for unique_key in util.unique_iterator(keys):
            group_data = dataset.select(**dict(zip(dimensions, unique_key)))
            if not len(group_data):
                continue
            group_data = group_type(group_data, **group_kwargs)
            grouped_data.append((unique_key, group_data))

        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(grouped_data, kdims=dimensions)
        else:
            return container_type(grouped_data)


    @classmethod
    def select_mask(cls, dataset, selection):
        """
        Given a Dataset object and a dictionary with dimension keys and
        selection keys (i.e tuple ranges, slices, sets, lists or literals)
        return a boolean mask over the rows in the Dataset object that
        have been selected.
        """
        mask = None
        for dim, sel in selection.items():
            if isinstance(sel, tuple):
                sel = slice(*sel)
            arr = cls.values(dataset, dim, compute=False)
            if util.isdatetime(arr) and util.pd:
                try:
                    sel = util.parse_datetime_selection(sel)
                except:
                    pass

            new_masks = []
            if isinstance(sel, slice):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'invalid value encountered')
                    if sel.start is not None:
                        new_masks.append(sel.start <= arr)
                    if sel.stop is not None:
                        new_masks.append(arr < sel.stop)
                new_mask = new_masks[0]
                for imask in new_masks[1:]:
                    new_mask &= imask
            elif isinstance(sel, (set, list)):
                for v in sel:
                    new_masks.append(arr==v)
                new_mask = new_masks[0]
                for imask in new_masks[1:]:
                    new_mask |= imask
            elif callable(sel):
                new_mask = sel(arr)
            else:
                new_mask = arr == sel

            if mask is None:
                mask = new_mask
            else:
                mask &= new_mask
        return mask


    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        df = dataset.data
        if selection_mask is None:
            selection_mask = cls.select_mask(dataset, selection)

        indexed = cls.indexed(dataset, selection)
        df = df[selection_mask]
        if indexed and len(df) == 1 and len(dataset.vdims) == 1:
            return df[dataset.vdims[0].name].iloc[0]
        return df


    @classmethod
    def concat_fn(cls, dataframes, **kwargs):
        import cudf
        return cudf.concat(dataframes, **kwargs)


    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        data = dataset.data
        cols = [d.name for d in dataset.kdims if d in dimensions]
        vdims = dataset.dimensions('value', label='name')
        reindexed = data[cols+vdims]
        agg = function.__name__
        if agg in ('amin', 'amax'):
            agg = agg[1:]
        if not hasattr(data, agg):
            raise ValueError('%s aggregation is not supported on cudf DataFrame.' % agg)
        if len(dimensions):
            grouped = reindexed.groupby(cols, sort=False)
            df = getattr(grouped, agg)().reset_index()
        else:
            agg = getattr(reindexed, agg)()
            data = dict(((col, [v]) for col, v in zip(agg.index, agg.to_array())))
            df = util.pd.DataFrame(data, columns=list(agg.index))

        dropped = []
        for vd in vdims:
            if vd not in df.columns:
                dropped.append(vd)
        return df, dropped


    @classmethod
    def sort(cls, dataset, by=[], reverse=False):
        dataset.param.warning("cuDF DataFrames do not yet support sorting.")
        return dataset.data

    @classmethod
    def dframe(cls, dataset, dimensions):
        if dimensions:
            return dataset.data[dimensions].to_pandas()
        else:
            return dataset.data.to_pandas()


Interface.register(cuDFInterface)
