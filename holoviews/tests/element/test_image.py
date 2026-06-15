"""
Unit tests of Image elements
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import holoviews as hv
from holoviews.testing import assert_element_equal

from ..utils import LoggingComparison


class TestImage(LoggingComparison):
    def setup_method(self):
        self.array1 = np.array([(0, 1, 2), (3, 4, 5)])

    def test_image_init(self):
        image = hv.Image(self.array1)
        assert image.xdensity == 3
        assert image.ydensity == 2

    def test_image_index(self):
        image = hv.Image(self.array1)
        assert image[-0.33, -0.25] == 3

    def test_image_sample(self):
        image = hv.Image(self.array1)
        assert_element_equal(
            image.sample(y=0.25),
            hv.Curve(np.array([(-0.333333, 0), (0, 1), (0.333333, 2)]), kdims=["x"], vdims=["z"]),
        )

    def test_image_range_masked(self):
        arr = np.random.rand(10, 10) - 0.5
        arr = np.ma.masked_where(arr <= 0, arr)
        rrange = hv.Image(arr).range(2)
        assert rrange == (np.min(arr), np.max(arr))

    def test_empty_image(self):
        hv.Image([])
        hv.Image(None)
        hv.Image(np.array([]))
        hv.Image(np.zeros((0, 0)))

    def test_image_rtol_failure(self):
        vals = np.random.rand(20, 20)
        xs = np.linspace(0, 10, 20)
        ys = np.linspace(0, 10, 20)
        ys[-1] += 0.1
        hv.Image({"vals": vals, "xs": xs, "ys": ys}, ["xs", "ys"], "vals")
        substr = (
            "set a higher tolerance on hv.config.image_rtol or "
            "the rtol parameter in the Image constructor."
        )
        self.log_handler.assert_endswith("WARNING", substr)

    def test_image_rtol_constructor(self):
        vals = np.random.rand(20, 20)
        xs = np.linspace(0, 10, 20)
        ys = np.linspace(0, 10, 20)
        ys[-1] += 0.01
        hv.Image({"vals": vals, "xs": xs, "ys": ys}, ["xs", "ys"], "vals", rtol=10e-2)

    def test_image_rtol_config(self):
        vals = np.random.rand(20, 20)
        xs = np.linspace(0, 10, 20)
        ys = np.linspace(0, 10, 20)
        ys[-1] += 0.001
        image_rtol = hv.config.image_rtol
        hv.config.image_rtol = 10e-3
        hv.Image({"vals": vals, "xs": xs, "ys": ys}, ["xs", "ys"], "vals")
        hv.config.image_rtol = image_rtol

    @pytest.mark.parametrize(
        "x",
        [
            # max(|step|) / min(|step|) - 1 = 0.0015
            [dt.datetime(2017, 1, 1, 0, 0, 0, ms) for ms in [0, 10000, 20015]],
            *(
                np.array([0, 10000, 20015]).astype(f"datetime64[{unit}]")
                for unit in ["D", "h", "m", "s", "ms", "us"]
            ),
        ],
    )
    def test_image_datetime_rtol_failure(self, x):
        """
        Constructing an Image with a date-time coordinate warns about irregular sampling
        if, approximately speaking, one less than the ratio of the max and min steps
        exceeds the default `image_rtol`.
        """
        y = np.arange(2)
        hv.Image((x, y, np.zeros((len(y), len(x)))))
        self.log_handler.assert_endswith(
            "WARNING",
            "set a higher tolerance on hv.config.image_rtol or the rtol parameter in "
            "the Image constructor.",
        )

    @pytest.mark.parametrize(
        "x",
        [
            # |granularity / step| = 1/400 = 0.0025
            [dt.datetime(2017, 1, 1, 0, 0, 0, ms) for ms in [0, 400]],
            # |granularity / step| = 1/800 = 0.00125
            *(
                np.array([10000, 10800]).astype(f"datetime64[{unit}]")
                for unit in ["D", "h", "m", "s", "ms", "us"]
            ),
        ],
    )
    def test_image_datetime_rtol_granularity(self, x):
        """
        Constructing an Image with an evenly spaced date-time coordinate does not warn
        about irregular sampling, even when the ratio of the date-time granularity to
        the coordinate step is greater than the default `image_rtol` (or twice
        `image_rtol` for `dt.datetime` types because of how `dt.timedelta` instances
        round when multiplied by a `float`).
        """
        y = np.arange(2)
        hv.Image((x, y, np.zeros((len(y), len(x)))))
        assert len(self.log_handler.tail("WARNING", n=1)) == 0

    def test_image_clone(self):
        vals = np.random.rand(20, 20)
        xs = np.linspace(0, 10, 20)
        ys = np.linspace(0, 10, 20)
        ys[-1] += 0.001
        img = hv.Image({"vals": vals, "xs": xs, "ys": ys}, ["xs", "ys"], "vals", rtol=10e-3)
        assert img.clone().rtol == 10e-3

    def test_image_curvilinear_coords_error(self):
        x = np.arange(-1, 1, 0.1)
        y = np.arange(-1, 1, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = np.sqrt(X**2 + Y**2) * np.cos(X)
        with pytest.raises(ValueError):  # noqa: PT011
            hv.Image((X, Y, Z))
