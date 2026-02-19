import pytest

import holoviews as hv

from .. import expect

pytestmark = pytest.mark.ui


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize("offset", ["xoffset", "yoffset"])
@pytest.mark.parametrize("with_scatter", [True, False])
def test_labels_offset(serve_hv, offset, with_scatter):
    labels = hv.Labels(
        {("x", "y"): [["a", 1], ["b", 2]], "text": ["a", "b"]},
        ["x", "y"],
        "text",
    ).opts(**{offset: 0.2})
    if with_scatter:
        scatter = hv.Scatter([("a", 1), ("b", 2)]).opts(padding=0.1)
        plot = labels * scatter
    else:
        plot = labels

    page = serve_hv(plot)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)
