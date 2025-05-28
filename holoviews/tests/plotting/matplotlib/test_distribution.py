import pytest

import holoviews as hv


@pytest.mark.usefixtures("mpl_backend")
def test_distribution_legend(rng):
    normal = rng.normal(1000)
    binary = rng.integers(0, 2, 1000)
    df = {"normal": normal, "binary": binary}

    ds = hv.Dataset(df, kdims=["binary"], vdims=["normal"])
    plot = ds.to(hv.Distribution, "normal").overlay("binary")
    assert "legend" in hv.renderer("matplotlib").get_plot(plot).handles
