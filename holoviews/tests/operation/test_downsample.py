import pytest

import holoviews as hv
from holoviews.operation.downsample import downsample1d


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
