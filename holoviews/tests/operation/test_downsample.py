import numpy as np
import pytest

import holoviews as hv
from holoviews.operation.downsample import _ALGORITHMS, downsample1d

try:
    import tsdownsample
except ImportError:
    tsdownsample = None

algorithms = _ALGORITHMS.copy()
algorithms.pop('viewport', None)  # viewport return slice(len(data)) no matter the width

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


@pytest.mark.skipif(not tsdownsample, reason="tsdownsample not installed")
@pytest.mark.parametrize("algorithm", algorithms.values(), ids=algorithms)
def test_downsample_algorithm_with_tsdownsample(algorithm):
    x = np.arange(1000)
    y = np.random.rand(1000)
    width = 20
    result = algorithm(x, y, width)
    if isinstance(result, slice):
        result = x[result]
    assert result.size == width
