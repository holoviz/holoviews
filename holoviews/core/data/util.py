import numpy as np

from .. import util


def finite_range(column, cmin, cmax):
    minf = np.isfinite(cmin)
    maxf = np.isfinite(cmax)
    if not minf or not maxf:
        column = column[np.isfinite(column)]
        cmin = cmin if minf else np.nanmin(column)
        cmax = cmax if maxf else np.nanmmax(column)
        if is_dask(column):
            import dask.array as da
            cmin, cmax = da.compute(cmin, cmax)
    if isinstance(cmin, np.ndarray) and cmin.shape == ():
        cmin = cmin[()]
    if isinstance(cmax, np.ndarray) and cmax.shape == ():
        cmax = cmax[()]
    cmin = cmin if np.isscalar(cmin) or isinstance(cmin, util.datetime_types) else cmin.item()
    cmax = cmax if np.isscalar(cmax) or isinstance(cmax, util.datetime_types) else cmax.item()
    return cmin, cmax

def get_array_types():
    array_types = (np.ndarray,)
    da = dask_array_module()
    if da is not None:
        array_types += (da.Array,)
    return array_types

def dask_array_module():
    try:
        import dask.array as da
        return da
    except:
        return None

def is_dask(array):
    da = dask_array_module()
    if da is None:
        return False
    return da and isinstance(array, da.Array)

def cached(method):
    """
    Decorates an Interface method and using a cached version
    """
    def cached(*args, **kwargs):
        cache = getattr(args[1], '_cached')
        if cache is None:
            return method(*args, **kwargs)
        else:
            args = (cache,)+args[2:]
            return getattr(cache.interface, method.__name__)(*args, **kwargs)
    return cached
