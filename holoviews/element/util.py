import numpy as np

from ..core import Dataset
from ..core.util import pd, is_nan

try:
    import dask
except:
    dask = None

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
    Generates a 2D aggregate by inserting NaNs at all cross-product
    locations that do not already have a value assigned, creating
    values that can be shaped into a 2D array.
    """
    if obj.interface.gridded:
        return obj
    elif obj.ndims > 2:
        raise Exception("Cannot aggregate more than two dimensions")
    d1keys = obj.dimension_values(0, False)
    d2keys = obj.dimension_values(1, False)
    coords = [(d1, d2) + (np.NaN,)*len(obj.vdims) for d1 in d1keys for d2 in d2keys]
    dtype = 'dataframe' if pd else 'dictionary'
    dense_data = Dataset(coords, kdims=obj.kdims, vdims=obj.vdims, datatype=[dtype])
    concat_data = obj.interface.concatenate([dense_data, Dataset(obj)], datatype=dtype)
    return concat_data.aggregate(obj.kdims, reduce_fn)
