import numpy as np
import pandas as pd
import pytest

import holoviews as hv


class TestDonutElement:
    def test_basic_construction(self):
        d = hv.Donut([("A", 30), ("B", 70)])
        assert d.kdims[0].name == "x"
        assert d.vdims[0].name == "y"
        assert len(d) == 2

    def test_construction_with_custom_dims(self):
        d = hv.Donut(
            [("Rent", 1200), ("Food", 400)],
            kdims=["Category"],
            vdims=["Amount"],
        )
        assert d.kdims[0].name == "Category"
        assert d.vdims[0].name == "Amount"

    def test_construction_from_dataframe(self):
        df = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [10, 20, 30]})
        d = hv.Donut(df, kdims="Category", vdims="Value")
        assert len(d) == 3
        np.testing.assert_equal(d.dimension_values("Value"), np.array([10, 20, 30]))

    @pytest.mark.parametrize(
        "data",
        [
            {"x": ["A", "B", "C"], "y": [10, 20, 30]},
            (["A", "B", "C"], [10, 20, 30]),
            [],
        ],
    )
    def test_construction_from_various_inputs(self, data):
        d = hv.Donut(data)
        expected_len = 0 if data == [] else 3
        assert len(d) == expected_len

    def test_group_name(self):
        assert hv.Donut([("A", 10)]).group == "Donut"

    def test_kdims_bounds(self):
        """Donut only accepts a single kdim."""
        with pytest.raises(ValueError, match="length must be between 1 and 1"):
            hv.Donut([("A", "x", 10)], kdims=["cat1", "cat2"], vdims=["val"])

    def test_dimension_values(self):
        d = hv.Donut([("A", 30), ("B", 70)])
        np.testing.assert_equal(np.asarray(d.dimension_values(0)), np.array(["A", "B"]))
        np.testing.assert_equal(d.dimension_values(1), np.array([30.0, 70.0]))

    def test_dframe(self):
        d = hv.Donut([("A", 30), ("B", 70)])
        df = d.dframe()
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 2
