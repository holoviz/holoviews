import numpy as np

try:
    import dask
except:
    dask = None

def toarray(v):
    """
    Interface helper function to turn dask Arrays into numpy arrays as
    necessary.
    """
    if dask and isinstance(v, dask.array.Array):
        return v.compute()
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
