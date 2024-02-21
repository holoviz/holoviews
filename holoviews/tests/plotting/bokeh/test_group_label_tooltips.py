import numpy as np
import pandas as pd

from holoviews.core import NdOverlay
from holoviews.element import Curve, Points

from .test_plot import TestBokehPlot


class TestGroupLabelPlot(TestBokehPlot):
    def test_group_label_overridden_by_data_dims(self):
        """test if the group and label args will be overridden by data dims of the same name"""
        group = "A"
        label = "B"
        data = pd.DataFrame(dict(group=range(4), Label=range(4)))
        obj = Curve(data, group=group, label=label).opts(tools=["hover"])
        tooltips = [
            ("group", "@{group}"),
            ("Label", "@{Label}"),
        ]
        self._test_hover_info(obj, tooltips, "nearest")

    def test_label_tooltip_from_arg(self):
        """test if the label arg will be added to the tooltips if not present in the data"""
        data = np.random.rand(10, 2)
        label = "B"
        obj = Points(data, kdims=["Group", "Dim2"], label=label).opts(tools=["hover"])
        tooltips = [
            ("Label", label),
            ("Group", "@{Group}"),
            ("Dim2", "@{Dim2}"),
        ]
        self._test_hover_info(obj, tooltips, "nearest")

    def test_group_not_overridden_by_similar_data_dim(self):
        """test if a data dim that has a matching part only will override group arg"""
        data = np.random.rand(10, 2)
        group = "A"
        label = "B"
        obj = Curve(data, kdims=["GroupA"], group=group, label=label).opts(
            tools=["hover"]
        )
        tooltips = [
            ("Group", group),
            ("Label", label),
            ("GroupA", "@{GroupA}"),
            ('y', '@{y}')
        ]
        self._test_hover_info(obj, tooltips, "nearest")

    def test_group_label_batched(self):
        """test if the group as a container dim will override the group arg on individual elements"""
        obj = NdOverlay(
            {
                i: Points(
                    [np.random.rand(10, 2)],
                    kdims=["Label", "Dim2"],
                    group="arg_test",
                    label="arg_test",
                )
                for i in range(5)
            },
            kdims=["group"],
        )
        opts = {"Points": {"tools": ["hover"]}, "NdOverlay": {"legend_limit": 0}}
        obj = obj.opts(opts)
        tooltips = [('group', '@{group}'), ('Label', '@{Label}'), ('Dim2', '@{Dim2}')]
        self._test_hover_info(obj, tooltips)
