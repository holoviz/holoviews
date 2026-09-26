from __future__ import annotations

import numpy as np
import pytest

import holoviews as hv
from holoviews.plotting.util import categorical_legend

from .._deps import ds

if not ds:
    pytest.skip("datashader not installed", allow_module_level=True)  # pragma: no cover

from holoviews.operation.datashader import datashade, rasterize

_CAT_KWARGS = dict(aggregator=ds.by("Label"), dynamic=False, width=10, height=10)


@pytest.fixture
def cat_points():
    return hv.Points([(0, 0, "A"), (1, 1, "B"), (2, 2, "C")], vdims=["Label"])


def test_categorical_legend_from_datashade(bokeh_backend, cat_points):
    rgb = datashade(cat_points, **_CAT_KWARGS)
    legend = categorical_legend(rgb, backend="bokeh")

    assert isinstance(legend, hv.Points)
    assert list(legend.dimension_values("category")) == ["A", "B", "C"]
    opts = legend.opts.get().kwargs
    assert opts["color"] == "category"
    assert opts["show_legend"] is True
    assert opts["visible"] is False
    assert set(opts["cmap"]) == {"A", "B", "C"}


def test_categorical_legend_from_rasterize(bokeh_backend, cat_points):
    img_stack = rasterize(cat_points, **_CAT_KWARGS)
    legend = categorical_legend(img_stack, backend="bokeh")

    assert isinstance(legend, hv.Points)
    assert list(legend.dimension_values("category")) == ["A", "B", "C"]
    assert set(legend.opts.get().kwargs["cmap"]) == {"A", "B", "C"}


def test_categorical_legend_from_count_cat(bokeh_backend, cat_points):
    rgb = datashade(
        cat_points, aggregator=ds.count_cat("Label"), dynamic=False, width=10, height=10
    )
    legend = categorical_legend(rgb, backend="bokeh")

    assert isinstance(legend, hv.Points)
    assert list(legend.dimension_values("category")) == ["A", "B", "C"]


def test_categorical_legend_uses_color_key(bokeh_backend, cat_points):
    color_key = {"A": "red", "B": "green", "C": "blue"}
    rgb = datashade(cat_points, color_key=color_key, **_CAT_KWARGS)
    legend = categorical_legend(rgb, backend="bokeh")

    assert legend.opts.get().kwargs["cmap"] == color_key


def test_categorical_legend_applies_cmap_list(bokeh_backend, cat_points):
    img_stack = rasterize(cat_points, **_CAT_KWARGS)
    legend = categorical_legend(img_stack, backend="bokeh", cmap=["red"])

    assert legend.opts.get().kwargs["cmap"] == {"A": "red", "B": "red", "C": "red"}


def test_categorical_legend_plain_rgb_returns_none(bokeh_backend):
    assert categorical_legend(hv.RGB(np.zeros((2, 2, 3))), backend="bokeh") is None


def test_categorical_legend_non_categorical_datashade_returns_none(bokeh_backend):
    rgb = datashade(hv.Points([(0, 0), (1, 1)]), dynamic=False, width=10, height=10)
    assert categorical_legend(rgb, backend="bokeh") is None


def test_categorical_legend_count_aggregator_returns_none(bokeh_backend, cat_points):
    rgb = datashade(cat_points, aggregator=ds.count(), dynamic=False, width=10, height=10)
    assert categorical_legend(rgb, backend="bokeh") is None
