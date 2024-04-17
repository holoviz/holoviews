import numpy as np
import pytest

import holoviews as hv

from .. import expect

pytestmark = pytest.mark.ui


@pytest.mark.usefixtures("bokeh_backend")
def test_gridspace_toolbar(serve_hv):
    def sine_curve(phase, freq):
        xvals = [0.1 * i for i in range(100)]
        return hv.Curve((xvals, [np.sin(phase + freq * x) for x in xvals]))

    phases = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    frequencies = [0.5, 0.75, 1.0, 1.25]
    curve_dict_2D = {(p, f): sine_curve(p, f) for p in phases for f in frequencies}
    gridspace = hv.GridSpace(curve_dict_2D, kdims=["phase", "frequency"])

    page = serve_hv(gridspace)
    bokeh_logo = page.locator(".bk-logo")
    expect(bokeh_logo).to_have_count(1)
