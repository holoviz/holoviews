from __future__ import absolute_import

from itertools import product

import numpy as np

from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap

try:
    from bokeh.models import ColorBar
    from holoviews.plotting.bokeh import RadialHeatMapPlot
except:
    pass

from .testplot import TestBokehPlot, bokeh_renderer


class BokehRadialHeatMapPlotTests(TestBokehPlot):

    def setUp(self):
        super(BokehRadialHeatMapPlotTests, self).setUp()
        # set up dummy data for convenient tests
        x = ["Seg {}".format(idx) for idx in range(2)]
        y = ["Ann {}".format(idx) for idx in range(2)]
        self.z = list(range(4))
        self.x, self.y = zip(*product(x, y))

        self.ann_bins = {"o1": np.array([0.5, 0.75]),
                         "o2": np.array([0.75, 1])}

        self.seg_bins = {"o1": np.array([0, np.pi]),
                         "o2": np.array([np.pi, 2*np.pi])}

        # set up plot options for convenient tests
        plot_opts = dict(start_angle=0,
                         max_radius=1,
                         radius_inner=0.5,
                         radius_outer=0.2,
                         radial=True)

        opts = dict(HeatMap=dict(plot=plot_opts))

        # provide element and plot instances for tests
        self.element = HeatMap((self.x, self.y, self.z)).opts(opts)
        self.plot = bokeh_renderer.get_plot(self.element)


    def test_radius_bin_computation(self):
        """Test computation of bins for radius/annulars.

        """

        order = sorted(self.ann_bins.keys())
        values = self.plot._get_bins("radius", order)
        self.assertEqual(values.keys(), self.ann_bins.keys())
        self.assertEqual(values["o1"], self.ann_bins["o1"])
        self.assertEqual(values["o2"], self.ann_bins["o2"])

        values = self.plot._get_bins("radius", order, reverse=True)
        self.assertEqual(values.keys(), self.ann_bins.keys())
        self.assertEqual(values["o1"], self.ann_bins["o2"])
        self.assertEqual(values["o2"], self.ann_bins["o1"])

    def test_angle_bin_computation(self):
        """Test computation of bins for radiants/segments.

        """
        order = sorted(self.seg_bins.keys())
        values = self.plot._get_bins("angle", order)
        self.assertEqual(values.keys(), self.seg_bins.keys())
        self.assertEqual(values["o1"], self.seg_bins["o1"])
        self.assertEqual(values["o2"], self.seg_bins["o2"])

        values = self.plot._get_bins("angle", order, True)
        self.assertEqual(values.keys(), self.seg_bins.keys())
        self.assertEqual(values["o1"], self.seg_bins["o2"])
        self.assertEqual(values["o2"], self.seg_bins["o1"])

    def test_plot_extents(self):
        """Test correct computation of extents.

        """

        extents = self.plot.get_extents("", "")
        self.assertEqual(extents, (-0.2, -0.2, 2.2, 2.2))

    def test_get_bounds(self):
        """Test boundary computation function.

        """

        order = ["o2", "o1", "o1"]
        start, end = self.plot._get_bounds(self.ann_bins, order)

        self.assertEqual(start, np.array([0.75, 0.5, 0.5]))
        self.assertEqual(end, np.array([1, 0.75, 0.75]))

    def test_compute_seg_tick_mappings(self):
        """Test computation of segment tick mappings. Check integers, list and
        function types.

        """
        order = sorted(self.seg_bins.keys())

        # test number ticks
        self.plot.xticks = 1
        ticks = self.plot._compute_tick_mapping("angle", order, self.seg_bins)
        self.assertEqual(ticks, {"o1": self.seg_bins["o1"]})

        self.plot.xticks = 2
        ticks = self.plot._compute_tick_mapping("angle", order, self.seg_bins)
        self.assertEqual(ticks, self.seg_bins)

        # test completely new ticks
        self.plot.xticks = ["New Tick1", "New Tick2"]
        ticks = self.plot._compute_tick_mapping("angle", order, self.seg_bins)
        bins = self.plot._get_bins("angle", self.plot.xticks, True)
        ticks_cmp = {x: bins[x] for x in self.plot.xticks}
        self.assertEqual(ticks, ticks_cmp)

        # test function ticks
        self.plot.xticks = lambda x: x == "o1"
        ticks = self.plot._compute_tick_mapping("angle", order, self.seg_bins)
        self.assertEqual(ticks, {"o1": self.seg_bins["o1"]})

    def test_compute_ann_tick_mappings(self):
        """Test computation of annular tick mappings. Check integers, list and
        function types.

        """

        order = sorted(self.ann_bins.keys())

        # test number ticks
        self.plot.yticks = 1
        ticks = self.plot._compute_tick_mapping("radius", order, self.ann_bins)
        self.assertEqual(ticks, {"o1": self.ann_bins["o1"]})

        self.plot.yticks = 2
        ticks = self.plot._compute_tick_mapping("radius", order, self.ann_bins)
        self.assertEqual(ticks, self.ann_bins)

        # test completely new ticks
        self.plot.yticks = ["New Tick1", "New Tick2"]
        ticks = self.plot._compute_tick_mapping("radius", order, self.ann_bins)
        bins = self.plot._get_bins("radius", self.plot.yticks)
        ticks_cmp = {x: bins[x] for x in self.plot.yticks}
        self.assertEqual(ticks, ticks_cmp)

        # test function ticks
        self.plot.yticks = lambda x: x == "o1"
        ticks = self.plot._compute_tick_mapping("radius", order, self.ann_bins)
        self.assertEqual(ticks, {"o1": self.ann_bins["o1"]})

    def test_get_default_mapping(self):
        # check for presence of glyphs in mapping
        # there is no sense in checking the default mapping here again
        # because it is basically a definition

        glyphs = self.plot._style_groups.keys()
        glyphs_mapped = self.plot.get_default_mapping(None, None).keys()
        glyphs_plain = set([x[:-2] for x in glyphs_mapped])

        self.assertTrue(all([x in glyphs_plain for x in glyphs]))

    def test_get_seg_labels_data(self):
        """Test correct computation of a single segment label data point.

        """

        radiant = np.pi / 2
        x = np.cos(radiant) + 1
        y = np.sin(radiant) + 1
        angle = 1.5 * np.pi + radiant

        test_seg_data = dict(x=np.array(x),
                             y=np.array(y),
                             text=np.array("o1"),
                             angle=np.array(angle))

        cmp_seg_data = self.plot._get_seg_labels_data(["o1"], self.seg_bins)

        self.assertEqual(test_seg_data, cmp_seg_data)

    def test_get_ann_labels_data(self):
        """Test correct computation of a single annular label data point.

        """

        test_ann_data = dict(x=np.array(1),
                             y=np.array(self.ann_bins["o1"][0]+1),
                             text=np.array("o1"),
                             angle=[0])

        cmp_ann_data = self.plot._get_ann_labels_data(["o1"], self.ann_bins)

        self.assertEqual(test_ann_data, cmp_ann_data)

    def test_get_markers(self):
        """Test computation of marker positions for function, list, tuple and
        integer type.

        """

        args = [sorted(self.ann_bins.keys()), self.ann_bins]

        # function type
        test_val = np.array(self.ann_bins["o1"][1])
        test_input = lambda x: x=="o1"
        self.assertEqual(self.plot._get_markers(test_input, *args), test_val)

        # list type
        test_val = np.array(self.ann_bins["o2"][1])
        test_input = [1]
        self.assertEqual(self.plot._get_markers(test_input, *args), test_val)

        # tuple type
        test_val = np.array(self.ann_bins["o2"][1])
        test_input = ("o2", )
        self.assertEqual(self.plot._get_markers(test_input, *args), test_val)

        # integer type
        test_val = np.array(self.ann_bins["o1"][1])
        test_input = 1
        self.assertEqual(self.plot._get_markers(test_input, *args), test_val)

    def test_get_xmarks_data(self):
        """Test computation of xmarks data for single xmark.

        """

        self.plot.xmarks = 1
        test_mark_data = dict(ys=[(1, 1)], xs=[(0.5, 0)])
        cmp_mark_data = self.plot._get_xmarks_data(["o1"], self.seg_bins)

        for sub_list in ["xs", "ys"]:
            test_pairs = zip(test_mark_data[sub_list], cmp_mark_data[sub_list])
            for test_value, cmp_value in test_pairs:
                self.assertEqual(np.array(test_value), np.array(cmp_value))


    def test_get_ymarks_data(self):
        """Test computation of ymarks data for single ymark.

        """

        self.plot.ymarks = 1
        test_mark_data = dict(radius=self.ann_bins["o1"][1])
        cmp_mark_data = self.plot._get_ymarks_data(["o1"], self.ann_bins)

        self.assertEqual(test_mark_data, cmp_mark_data)

    def test_get_data(self):
        """Test for presence of glyphs in data and mapping. Testing for correct
        values for data and mapping are covered via other unit tests and hence
        are skipped here.

        """

        data, mapping, style = self.plot.get_data(self.element, {}, {})

        glyphs = self.plot._style_groups.keys()

        for check in [data, mapping]:
            glyphs_mapped = check.keys()
            glyphs_plain = set([x[:-2] for x in glyphs_mapped])
            self.assertTrue(all([x in glyphs_plain for x in glyphs]))

    def test_plot_data_source(self):
        """Test initialization of ColumnDataSources.

        """

        source_ann = self.plot.handles['annular_wedge_1_source'].data

        self.assertEqual(list(source_ann["x"]), list(self.x))
        self.assertEqual(list(source_ann["y"]), list(self.y))
        self.assertEqual(list(source_ann["z"]), self.z)

    def test_heatmap_holomap(self):
        hm = HoloMap({'A': HeatMap(np.random.randint(0, 10, (100, 3))),
                      'B': HeatMap(np.random.randint(0, 10, (100, 3)))})
        plot = bokeh_renderer.get_plot(hm.options(radial=True))
        self.assertIsInstance(plot, RadialHeatMapPlot)

    def test_radial_heatmap_colorbar(self):
        hm = HeatMap([(0, 0, 1), (0, 1, 2), (1, 0, 3)]).options(radial=True, colorbar=True)
        plot = bokeh_renderer.get_plot(hm)
        self.assertIsInstance(plot.handles.get('colorbar'), ColorBar)

    def test_radial_heatmap_ranges(self):
        hm = HeatMap([(0, 0, 1), (0, 1, 2), (1, 0, 3)]).options(radial=True, colorbar=True)
        plot = bokeh_renderer.get_plot(hm)
        self.assertEqual(plot.handles['x_range'].start, -0.05)
        self.assertEqual(plot.handles['x_range'].end, 1.05)
        self.assertEqual(plot.handles['y_range'].start, -0.05)
        self.assertEqual(plot.handles['y_range'].end, 1.05)
