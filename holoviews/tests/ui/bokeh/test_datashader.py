from functools import partial
from unittest import SkipTest

import pandas as pd
import pytest

import holoviews as hv
from holoviews.operation import apply_when

try:
    from holoviews.operation.datashader import rasterize
except ImportError:
    raise SkipTest("Datashader not available")
from ...plotting.utils import ParamLogStream
from .. import expect, wait_until

pytestmark = pytest.mark.ui


@pytest.mark.usefixtures("bokeh_backend")
def test_apply_when_zoom(serve_hv):
    df = pd.DataFrame(
        {
            "lon": [45.5531, 46.4731, 47.4731, 48.41234],
            "lat": [34.5531, 35.4731, 36.4731, 37.41234],
        }
    )

    def _resample_obj(operation, obj, opts):
        def exceeds_resample_when(plot):
            return len(plot) > 1000

        processed = apply_when(
            obj, operation=partial(operation, **opts), predicate=exceeds_resample_when
        )
        return processed

    points = hv.Points(df, ["lon", "lat"])
    resampled_points = _resample_obj(rasterize, points, {})

    page = serve_hv(resampled_points)

    hv_plot = page.locator(".bk-events")
    wait_until(lambda: expect(hv_plot).to_have_count(1), page=page)
    bbox = hv_plot.bounding_box()

    # Hover over the plot
    page.mouse.move(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    hv_plot.click()

    with ParamLogStream() as log:
        page.mouse.wheel(0, 100)  # Scroll up (zoom in)
    log_msg = log.stream.read()
    assert not log_msg
