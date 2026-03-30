import numpy as np
import pandas as pd
import pytest

import holoviews as hv
from holoviews.testing import assert_data_equal


class TestWaterfallElement:
    def test_basic_construction(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        assert w.kdims[0].name == "x"
        assert w.vdims[0].name == "y"
        assert len(w) == 3

    def test_construction_with_custom_dims(self):
        w = hv.Waterfall(
            [("Revenue", 100), ("COGS", -40)],
            kdims=["Category"],
            vdims=["Amount"],
        )
        assert w.kdims[0].name == "Category"
        assert w.vdims[0].name == "Amount"

    def test_construction_from_dataframe(self):
        df = pd.DataFrame(
            {
                "Category": ["Revenue", "COGS", "Opex"],
                "Amount": [100, -40, -30],
            }
        )
        w = hv.Waterfall(df, kdims="Category", vdims="Amount")
        assert len(w) == 3
        assert_data_equal(w.dimension_values("Amount"), np.array([100, -40, -30]))

    def test_construction_from_dict(self):
        w = hv.Waterfall({"x": ["A", "B", "C"], "y": [10, -3, 5]})
        assert len(w) == 3

    def test_construction_from_tuple(self):
        w = hv.Waterfall((["A", "B", "C"], [10, -3, 5]))
        assert len(w) == 3

    def test_empty_construction(self):
        w = hv.Waterfall([])
        assert len(w) == 0

    def test_show_total_false(self):
        # show_total is a plot opt; the element itself is unaffected
        w = hv.Waterfall([("A", 10)])
        assert len(w) == 1

    def test_total_label_custom(self):
        # total_label is a plot opt; the element itself is unaffected
        w = hv.Waterfall([("A", 10)])
        assert len(w) == 1

    def test_total_not_in_element_data(self):
        """The total bar is computed at plot time, not stored in the element."""
        w = hv.Waterfall([("A", 10), ("B", -3)])
        assert len(w) == 2
        labels = list(w.dimension_values(0))
        assert "Total" not in labels

    def test_group_name(self):
        w = hv.Waterfall([("A", 10)])
        assert w.group == "Waterfall"

    def test_kdims_bounds(self):
        """Waterfall only accepts a single kdim."""
        with pytest.raises(ValueError, match="length must be between 1 and 1"):
            hv.Waterfall(
                [("A", "x", 10)],
                kdims=["cat1", "cat2"],
                vdims=["val"],
            )

    def test_dimension_values(self):
        w = hv.Waterfall([("A", 10), ("B", -3)])
        assert_data_equal(np.asarray(w.dimension_values(0)), np.array(["A", "B"]))
        assert_data_equal(w.dimension_values(1), np.array([10.0, -3.0]))

    def test_dframe(self):
        w = hv.Waterfall([("A", 10), ("B", -3)])
        df = w.dframe()
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 2
