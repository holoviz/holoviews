from __future__ import annotations

import numpy as np
import pytest

from holoviews.core.data.util import finite_range

from ..._deps import cp, da, da_skip, xr, xr_skip


@xr_skip
@pytest.mark.parametrize(
    "xp",
    [np, pytest.param(cp, marks=pytest.mark.gpu), pytest.param(da, marks=da_skip)],
)
def test_finite_range_on_nonfinite_2D_xarray(xp):
    pos_inf = xr.DataArray(xp.array([[0, 1, 2], [3, np.inf, 5]]))
    assert finite_range(pos_inf, -np.inf, np.inf) == (0, 5)
    neg_inf = xr.DataArray(xp.array([[0, 1, -np.inf], [3, 4, 5]]))
    assert finite_range(neg_inf, -np.inf, np.inf) == (0, 5)
    both_inf = xr.DataArray(xp.array([[-np.inf, 1, 2], [3, 4, np.inf]]))
    assert finite_range(both_inf, -np.inf, np.inf) == (1, 4)
