import sys

import numpy as np

def get_array_types():
    array_types = (np.ndarray,)
    if 'dask' in sys.modules:
        import dask.array as da
        array_types += (da.Array,)
    return array_types

def get_dask_array():
    try:
        import dask.array as da
        return da
    except:
        return None

def is_dask(array):
    if 'dask' in sys.modules:
        import dask.array as da
    else:
        return False
    return da and isinstance(array, da.Array)
