from typing import TYPE_CHECKING

import numpy as np

from .. import util

if TYPE_CHECKING:
    import dask.array as da
else:
    da = util._LazyModule("dask.array", bool_use_sys_modules=True)


def finite_range(column, cmin, cmax):
    try:
        min_inf = np.isinf(cmin)
    except TypeError:
        min_inf = False
    try:
        max_inf = np.isinf(cmax)
    except TypeError:
        max_inf = False
    if (min_inf or max_inf):
        column = column[np.isfinite(column)]
        if len(column):
            cmin = np.nanmin(column) if min_inf else cmin
            cmax = np.nanmax(column) if max_inf else cmax
            if is_dask(column):
                import dask.array as da
                if min_inf and max_inf:
                    cmin, cmax = da.compute(cmin, cmax)
                elif min_inf:
                    cmin = cmin.compute()
                else:
                    cmax = cmax.compute()
    else:
        return cmin, cmax
    if isinstance(cmin, np.ndarray) and cmin.shape == ():
        cmin = cmin[()]
    if isinstance(cmax, np.ndarray) and cmax.shape == ():
        cmax = cmax[()]
    cmin = cmin if np.isscalar(cmin) or isinstance(cmin, util.datetime_types) else cmin.item()
    cmax = cmax if np.isscalar(cmax) or isinstance(cmax, util.datetime_types) else cmax.item()
    return cmin, cmax

def get_array_types():
    array_types = (np.ndarray,)
    if da:
        array_types += (da.Array,)
    return array_types

def dask_array_module():
    return da if da else None

def is_dask(array):
    return da and isinstance(array, da.Array)

def cached(method):
    """Decorates an Interface method and using a cached version

    """
    def cached(*args, **kwargs):
        cache = args[1]._cached
        if cache is None:
            return method(*args, **kwargs)
        else:
            args = (cache, *args[2:])
            return getattr(cache.interface, method.__name__)(*args, **kwargs)
    return cached
