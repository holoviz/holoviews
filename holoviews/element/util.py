import numpy as np

from ..core import Dataset, OrderedDict
from ..core.util import pd, is_nan

try:
    import dask
except:
    dask = None

try:
    import xarray as xr
except:
    xr = None


def toarray(v, index_value=False):
    """
    Interface helper function to turn dask Arrays into numpy arrays as
    necessary. If index_value is True, a value is returned instead of
    an array holding a single value.
    """
    if dask and isinstance(v, dask.array.Array):
        arr =  v.compute()
        return arr[()] if index_value else arr
    else:
        return v

def compute_edges(edges):
    """
    Computes edges from a number of bin centers,
    throwing an exception if the edges are not
    evenly spaced.
    """
    widths = np.diff(edges)
    if np.allclose(widths, widths[0]):
        width = widths[0]
    else:
        raise ValueError('Centered bins have to be of equal width.')
    edges -= width/2.
    return np.concatenate([edges, [edges[-1]+width]])


def reduce_fn(x):
    """
    Aggregation function to get the first non-zero value.
    """
    values = x.values if pd and isinstance(x, pd.Series) else x
    for v in values:
        if not is_nan(v):
            return v
    return np.NaN


def get_2d_aggregate(obj):
    """
    Generates a categorical 2D aggregate by inserting NaNs at all
    cross-product locations that do not already have a value assigned.
    Returns a 2D gridded Dataset object.
    """
    if obj.interface.gridded:
        return obj
    elif obj.ndims > 2:
        raise Exception("Cannot aggregate more than two dimensions")

    dims = obj.dimensions(label=True)
    xdim, ydim = dims[:2]
    nvdims = len(dims) - 2
    d1keys = obj.dimension_values(xdim, False)
    d2keys = obj.dimension_values(ydim, False)

    is_sorted = np.array_equal(np.sort(d1keys), d1keys)
    if is_sorted:
        grouped = obj.groupby(xdim, container_type=OrderedDict,
                              group_type=Dataset).values()
        for group in grouped:
            d2vals = group.dimension_values(ydim)
            is_sorted &= np.array_equal(d2vals, np.sort(d2vals))

    if is_sorted:
        d1keys, d2keys = np.sort(d1keys), np.sort(d2keys)
    coords = [(d1, d2) + (np.NaN,)*nvdims for d2 in d2keys for d1 in d1keys]

    dtype = 'dataframe' if pd else 'dictionary'
    dense_data = Dataset(coords, kdims=obj.kdims, vdims=obj.vdims, datatype=[dtype])
    concat_data = obj.interface.concatenate([dense_data, Dataset(obj)], datatype=dtype)
    agg = concat_data.reindex([xdim, ydim]).aggregate([xdim, ydim], reduce_fn)
    shape = (len(d2keys), len(d1keys))
    grid_data = {xdim: d1keys, ydim: d2keys}

    for vdim in dims[2:]:
        data = agg.dimension_values(vdim).reshape(shape)
        data = np.ma.array(data, mask=np.logical_not(np.isfinite(data)))
        grid_data[vdim] = data

    grid_type = 'xarray' if xr else 'grid'
    return agg.clone(grid_data, datatype=[grid_type])

