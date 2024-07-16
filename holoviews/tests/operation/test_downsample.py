import numpy as np
import pandas as pd
import pytest

from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.operation.downsample import _ALGORITHMS, downsample1d

try:
    import tsdownsample
except ImportError:
    tsdownsample = None

algorithms = _ALGORITHMS.copy()
algorithms.pop("viewport", None)  # viewport return slice(len(data)) no matter the width


@pytest.mark.parametrize("plottype", ["overlay", "ndoverlay"])
def test_downsample1d_multi(plottype):
    N = 1000
    assert N > downsample1d.width

    if plottype == "overlay":
        figure = Overlay([Curve(range(N)), Curve(range(N))])
    elif plottype == "ndoverlay":
        figure = NdOverlay({"A": Curve(range(N)), "B": Curve(range(N))})

    figure_values = downsample1d(figure, dynamic=False).data.values()
    for n in figure_values:
        for value in n.data.values():
            assert value.size == downsample1d.width


@pytest.mark.skipif(not tsdownsample, reason="tsdownsample not installed")
@pytest.mark.parametrize("algorithm", algorithms)
def test_downsample1d_non_contiguous(algorithm):
    x = np.arange(20)
    y = np.arange(40).reshape(1, 40)[0, ::2]

    downsampled = downsample1d(Curve((x, y), datatype=['array']), dynamic=False, width=10, algorithm=algorithm)
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
    figure = Overlay([Curve(df, kdims="x", vdims=c) for c in "yz"])

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
    figure = Overlay([Curve(df, kdims="index", vdims=c) for c in "xyz"])

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
