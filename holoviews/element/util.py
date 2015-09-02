import numpy as np

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
