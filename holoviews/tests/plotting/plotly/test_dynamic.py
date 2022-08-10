from unittest.mock import Mock

import holoviews as hv
import panel as pn
import numpy as np

from holoviews.streams import (
    Stream, Selection1D, RangeXY, BoundsXY,
)

from bokeh.document import Document
from pyviz_comms import Comm

from .test_plot import TestPlotlyPlot


class TestDynamicMap(TestPlotlyPlot):

    def test_update_dynamic_map_with_stream(self):
        ys = np.arange(10)

        # Build stream
        Scale = Stream.define('Scale', scale=1.0)
        scale_stream = Scale()

        # Build DynamicMap
        def build_scatter(scale):
            return hv.Scatter(ys * scale)

        dmap = hv.DynamicMap(build_scatter, streams=[scale_stream])

        # Create HoloViews Pane using panel so that we can access the plotly pane
        # used to display the plotly figure
        dmap_pane = pn.pane.HoloViews(dmap, backend='plotly')

        # Call get_root to force instantiation of internal plots/models
        doc = Document()
        comm = Comm()
        dmap_pane.get_root(doc, comm)

        # Get reference to the plotly pane
        _, plotly_pane = next(iter(dmap_pane._plots.values()))

        # Check initial data
        data = plotly_pane.object['data']
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['type'], 'scatter')
        np.testing.assert_equal(data[0]['y'], ys)

        # Watch object for changes
        fn = Mock()
        plotly_pane.param.watch(fn, 'object')

        # Update stream
        scale_stream.event(scale=2.0)

        # Check that figure object was updated
        data = plotly_pane.object['data']
        np.testing.assert_equal(data[0]['y'], ys * 2.0)

        # Check that object callback was triggered
        fn.assert_called_once()
        args, kwargs = fn.call_args_list[0]
        event = args[0]
        self.assertIs(event.obj, plotly_pane)
        self.assertIs(event.new, plotly_pane.object)


class TestInteractiveStream(TestPlotlyPlot):
    # Note: Testing the core logic of each interactive stream should take place in
    # testcallbacks.py. Here we are testing that that callbacks are properly
    # routed to streams

    def test_interactive_streams(self):
        ys = np.arange(10)
        scatter1 = hv.Scatter(ys)
        scatter2 = hv.Scatter(ys)
        scatter3 = hv.Scatter(ys)

        # Single stream on the first scatter
        rangexy1 = RangeXY(source=scatter1)

        # Multiple streams of the same type on second scatter
        boundsxy2a = BoundsXY(source=scatter2)
        boundsxy2b = BoundsXY(source=scatter2)

        # Multiple streams of different types on third scatter
        rangexy3 = RangeXY(source=scatter3)
        boundsxy3 = BoundsXY(source=scatter3)
        selection1d3 = Selection1D(source=scatter3)

        # Build layout and layout Pane
        layout = scatter1 + scatter2 + scatter3
        layout_pane = pn.pane.HoloViews(layout, backend='plotly')

        # Get plotly pane reference
        doc = Document()
        comm = Comm()
        layout_pane.get_root(doc, comm)
        _, plotly_pane = next(iter(layout_pane._plots.values()))

        # Simulate zoom and check that RangeXY streams updated accordingly
        plotly_pane.viewport = {
            'xaxis.range': [1, 3],
            'yaxis.range': [2, 4],
            'xaxis2.range': [3, 5],
            'yaxis2.range': [4, 6],
            'xaxis3.range': [5, 7],
            'yaxis3.range': [6, 8],
        }

        self.assertEqual(rangexy1.x_range, (1, 3))
        self.assertEqual(rangexy1.y_range, (2, 4))
        self.assertEqual(rangexy3.x_range, (5, 7))
        self.assertEqual(rangexy3.y_range, (6, 8))

        plotly_pane.viewport = None
        self.assertIsNone(rangexy1.x_range)
        self.assertIsNone(rangexy1.y_range)
        self.assertIsNone(rangexy3.x_range)
        self.assertIsNone(rangexy3.y_range)

        # Simulate box selection and check that BoundsXY and Selection1D streams
        # update accordingly

        # Box select on second subplot
        plotly_pane.selected_data = {
            'points': [],
            'range': {
                'x2': [10, 20],
                'y2': [11, 22]
            }
        }

        self.assertEqual(boundsxy2a.bounds, (10, 11, 20, 22))
        self.assertEqual(boundsxy2b.bounds, (10, 11, 20, 22))

        # Box selecrt on third subplot
        plotly_pane.selected_data = {
            'points': [
                {'curveNumber': 2, 'pointNumber': 0},
                {'curveNumber': 2, 'pointNumber': 3},
                {'curveNumber': 2, 'pointNumber': 7},
            ],
            'range': {
                'x3': [0, 5],
                'y3': [1, 6]
            }
        }

        self.assertEqual(boundsxy3.bounds, (0, 1, 5, 6))
        self.assertEqual(selection1d3.index, [0, 3, 7])

        # bounds streams on scatter 2 are None
        self.assertIsNone(boundsxy2a.bounds)
        self.assertIsNone(boundsxy2b.bounds)

        # Clear selection
        plotly_pane.selected_data = None
        self.assertIsNone(boundsxy3.bounds)
        self.assertIsNone(boundsxy2a.bounds)
        self.assertIsNone(boundsxy2b.bounds)
        self.assertEqual(selection1d3.index, [])
