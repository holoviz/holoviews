import numpy as np

from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.element import Curve, Image
from holoviews.streams import Stream

from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer


class TestLayoutPlot(LoggingComparisonTestCase, TestMPLPlot):

    def test_layout_instantiate_subplots(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = mpl_renderer.get_plot(layout)
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)
        for i, pos in enumerate(positions):
            adjoint = plot.subplots[pos]
            if 'main' in adjoint.subplots:
                self.assertEqual(adjoint.subplots['main'].layout_num, i+1)

    def test_layout_empty_subplots(self):
        layout = Curve(range(10)) + NdOverlay() + HoloMap() + HoloMap({1: Image(np.random.rand(10,10))})
        plot = mpl_renderer.get_plot(layout)
        self.assertEqual(len(plot.subplots.values()), 2)
        self.log_handler.assertContains('WARNING', 'skipping subplot')
        self.log_handler.assertContains('WARNING', 'skipping subplot')

    def test_layout_instantiate_subplots_transposed(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = mpl_renderer.get_plot(layout.opts(transpose=True))
        positions = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)
        nums = [1, 5, 2, 3, 4]
        for pos, num in zip(positions, nums, strict=None):
            adjoint = plot.subplots[pos]
            if 'main' in adjoint.subplots:
                self.assertEqual(adjoint.subplots['main'].layout_num, num)

    def test_layout_dimensioned_stream_title_update(self):
        stream = Stream.define('Test', test=0)()
        dmap = DynamicMap(lambda test: Curve([]), kdims=['test'], streams=[stream])
        layout = dmap + Curve([])
        plot = mpl_renderer.get_plot(layout)
        self.assertIn('test: 0', plot.handles['title'].get_text())
        stream.event(test=1)
        self.assertIn('test: 1', plot.handles['title'].get_text())
        plot.cleanup()
        self.assertEqual(stream._subscribers, [])

    def test_layout_shared_axes_disabled(self):
        from holoviews.plotting.mpl import CurvePlot
        layout = (Curve([1, 2, 3]) + Curve([10, 20, 30])).opts(shared_axes=False)
        plot = mpl_renderer.get_plot(layout)
        cp1, cp2 = plot.traverse(lambda x: x, [CurvePlot])
        self.assertTrue(cp1.handles['axis'].get_ylim(), (1, 3))
        self.assertTrue(cp2.handles['axis'].get_ylim(), (10, 30))

    def test_layout_sublabel_offset(self):
        from holoviews.plotting.mpl import CurvePlot
        layout = Curve([]) + Curve([]) + Curve([]) + Curve([])
        layout.opts(sublabel_offset=1)
        plot = mpl_renderer.get_plot(layout)
        cps = plot.traverse(lambda x: x, [CurvePlot])
        assert cps[0].handles["sublabel"].get_text() == "B"
        assert cps[1].handles["sublabel"].get_text() == "C"
        assert cps[2].handles["sublabel"].get_text() == "D"
        assert cps[3].handles["sublabel"].get_text() == "E"

    def test_layout_sublabel_skip(self):
        from holoviews.plotting.mpl import CurvePlot
        layout = Curve([]) + Curve([]) + Curve([]) + Curve([])
        layout.opts(sublabel_skip=[1, 3])
        plot = mpl_renderer.get_plot(layout)
        cps = plot.traverse(lambda x: x, [CurvePlot])
        assert "sublabel" not in cps[0].handles
        assert cps[1].handles["sublabel"].get_text() == "A"
        assert "sublabel" not in cps[2].handles
        assert cps[3].handles["sublabel"].get_text() == "B"
