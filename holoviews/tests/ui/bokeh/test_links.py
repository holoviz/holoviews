from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import holoviews as hv
from holoviews.plotting.links import RangeToolLink

from .. import expect

pytestmark = pytest.mark.ui


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    ["index", "intervalsx", "x_range_src", "x_range_tgt"],
    [
        (
            range(3000),
            (100, 365),
            (0, 365),
            (0, 3000 - 1),
        ),
        (
            pd.date_range("2000-03-01", periods=3000),
            (timedelta(days=100), timedelta(days=365)),
            (
                np.array(["2000-03-01"], dtype="datetime64[ns]")[0],
                pd.Timestamp("2001-03-01"),
            ),
            np.array(["2000-03-01", "2008-05-17"], dtype="datetime64[ns]"),
        ),
    ],
    ids=["int", "datetime"],
)
def test_rangetool_link_interval(serve_hv, index, intervalsx, x_range_src, x_range_tgt):
    df = pd.DataFrame(range(3000), columns=["close"], index=index)
    df.index.name = "Date"

    aapl_curve = hv.Curve(df, "Date", ("close", "Price ($)"))
    tgt = aapl_curve.relabel("AAPL close price").opts(width=800, labelled=["y"])
    src = aapl_curve.opts(width=800, height=100, yaxis=None)

    RangeToolLink(src, tgt, axes=["x", "y"], intervalsx=intervalsx)
    layout = (tgt + src).cols(1)
    layout.opts(hv.opts.Layout(shared_axes=False))

    page = serve_hv(layout)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(2)

    bk_model = hv.render(layout)
    bk_src = bk_model.children[0][0]
    np.testing.assert_equal((bk_src.x_range.start, bk_src.x_range.end), x_range_src)
    bk_tgt = bk_model.children[1][0]
    np.testing.assert_equal((bk_tgt.x_range.start, bk_tgt.x_range.end), x_range_tgt)
