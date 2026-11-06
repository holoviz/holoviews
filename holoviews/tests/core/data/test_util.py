from __future__ import annotations

import numpy as np

from holoviews.core.data.util import finite_range

from ..._deps import xr, xr_skip


@xr_skip
def test_finite_range_on_nonfinite_2D_xarray():
    """finite_range must not crash on 2D xarray input with non-finite values,
    and results must be computed from the finite values, only.
    Regression test for https://github.com/holoviz/holoviews/issues/7140
    """
    pos_inf = xr.DataArray([[0, 1, 2], [3, np.inf, 5]])
    assert finite_range(pos_inf, -np.inf, np.inf) == (0, 5)
    neg_inf = xr.DataArray([[0, 1, -np.inf], [3, 4, 5]])
    assert finite_range(neg_inf, -np.inf, np.inf) == (0, 5)
    both_inf = xr.DataArray([[-np.inf, 1, 2], [3, 4, np.inf]])
    assert finite_range(both_inf, -np.inf, np.inf) == (1, 4)
