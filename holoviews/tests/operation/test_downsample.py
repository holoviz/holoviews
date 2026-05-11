from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import holoviews as hv
from holoviews.operation.downsample import _ALGORITHMS, downsample1d

from ..utils import tsdownsample_skip

algorithms = _ALGORITHMS.copy()
algorithms.pop("viewport", None)  # viewport return slice(len(data)) no matter the width


@pytest.mark.parametrize("plottype", ["overlay", "ndoverlay"])
def test_downsample1d_multi(plottype):
    N = 1000
    assert N > downsample1d.width

    if plottype == "overlay":
        figure = hv.Overlay([hv.Curve(range(N)), hv.Curve(range(N))])
    elif plottype == "ndoverlay":
        figure = hv.NdOverlay({"A": hv.Curve(range(N)), "B": hv.Curve(range(N))})

    figure_values = downsample1d(figure, dynamic=False).data.values()
    for n in figure_values:
        for value in n.data.values():
            assert value.size == downsample1d.width


@tsdownsample_skip
@pytest.mark.parametrize("algorithm", algorithms)
def test_downsample1d_non_contiguous(algorithm):
    x = np.arange(20)
    y = np.arange(40).reshape(1, 40)[0, ::2]

    downsampled = downsample1d(
        hv.Curve((x, y), datatype=["array"]), dynamic=False, width=10, algorithm=algorithm
    )
    assert len(downsampled)


def test_downsample1d_shared_data():
    runs = [0]

    class mocksample(downsample1d):
        def _compute_mask(self, element):
            # Use _compute_mask as this should only be called once
            # and then it should be cloned.
            runs[0] += 1
            return super()._compute_mask(element)

    N = 1000
    df = pd.DataFrame({c: range(N) for c in "xyz"})
    figure = hv.Overlay([hv.Curve(df, kdims="x", vdims=c) for c in "yz"])

    # We set x_range to trigger _compute_mask
    mocksample(figure, dynamic=False, x_range=(0, 500))
    assert runs[0] == 1


def test_downsample1d_shared_data_index():
    runs = [0]

    class mocksample(downsample1d):
        def _compute_mask(self, element):
            # Use _compute_mask as this should only be called once
            # and then it should be cloned.
            runs[0] += 1
            return super()._compute_mask(element)

    N = 1000
    df = pd.DataFrame({c: range(N) for c in "xyz"})
    figure = hv.Overlay([hv.Curve(df, kdims="index", vdims=c) for c in "xyz"])

    # We set x_range to trigger _compute_mask
    mocksample(figure, dynamic=False, x_range=(0, 500))
    assert runs[0] == 1


@pytest.mark.parametrize("algorithm", algorithms.values(), ids=algorithms)
def test_downsample_algorithm(algorithm, unimport):
    unimport("tsdownsample")
    x = np.arange(1000)
    y = np.random.rand(1000)
    width = 20
    try:
        result = algorithm(x, y, width)
    except NotImplementedError:
        pytest.skip("not testing tsdownsample algorithms")
    else:
        if isinstance(result, slice):
            result = x[result]
        assert result.size == width


@tsdownsample_skip
@pytest.mark.parametrize("algorithm", algorithms.values(), ids=algorithms)
def test_downsample_algorithm_with_tsdownsample(algorithm):
    x = np.arange(1000)
    y = np.random.rand(1000)
    width = 20
    result = algorithm(x, y, width)
    if isinstance(result, slice):
        result = x[result]
    assert result.size == width
