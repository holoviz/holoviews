from holoviews.core import HoloMap, NdOverlay, Overlay, GridSpace, DynamicMap
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Stream, PlotSize, RangeXY
from holoviews.operation.datashader import spread, datashade
import param


class XY(Stream):
    x = param.Number(constant=True)
    y = param.Number(constant=True)


class Z(Stream):
    z = param.Number(constant=True)


class PX(Stream):
    px = param.Integer(constant=True)


class TestCollation(ComparisonTestCase):
    def setUp(self):
        from holoviews.tests.teststreams import Sum, Val

        # kdims: a and b
        self.dmap_ab = DynamicMap(
            lambda a, b: Points([a, b]),
            kdims=['a', 'b']
        ).redim.range(a=(0.0, 10.0), b=(0.0, 10.0))

        # kdims: b
        self.dmap_b = DynamicMap(
            lambda b: Points([b, b]),
            kdims=['b']
        ).redim.range(b=(0.0, 10.0))

        # no kdims, XY stream
        self.xy_stream = XY()
        self.dmap_xy = DynamicMap(
            lambda x, y: Points([x, y]), streams=[self.xy_stream]
        )

        # no kdims, Z stream
        self.z_stream = Z()
        self.dmap_z = DynamicMap(
            lambda z: Points([z, z]), streams=[self.z_stream]
        )

        # dmap produced by chained datashade and shade
        self.px_stream = PX()
        self.dmap_spread_points = spread(
            datashade(Points([0.0, 1.0])), streams=[self.px_stream]
        )

        # data shaded with kdims: a, b
        self.dmap_datashade_kdim_points = datashade(self.dmap_ab)

        # DynamicMap of a derived stream
        self.stream_val1 = Val()
        self.stream_val2 = Val()
        self.stream_val3 = Val()
        self.dmap_derived = DynamicMap(
            lambda v: Points([v, v]),
            streams=[
                Sum([self.stream_val1, Sum([self.stream_val2, self.stream_val3])])
            ]
        )

    def test_uncollate_layout_kdims(self):
        layout = self.dmap_ab + self.dmap_b
        uncollated = layout.uncollate()
        self.assertIsInstance(uncollated, DynamicMap)
        self.assertEqual(uncollated.kdims, self.dmap_ab.kdims)
        self.assertEqual(
            uncollated[2, 3],
            Points([2, 3]) + Points([3, 3])
        )
        self.assertEqual(
            uncollated.callback.callable(2, 3),
            Points([2, 3]) + Points([3, 3])
        )

    def test_uncollate_layout_streams(self):
        layout = self.dmap_xy + self.dmap_z
        uncollated = layout.uncollate()
        self.assertIsInstance(uncollated, DynamicMap)
        self.assertEqual(uncollated.kdims, [])

        # Update streams
        uncollated.streams[0].event(x=1.0, y=2.0)
        uncollated.streams[1].event(z=3.0)
        self.assertEqual(
            uncollated[()],
            Points([1.0, 2.0]) + Points([3.0, 3.0])
        )
        self.assertEqual(
            uncollated.callback.callable(dict(x=1.0, y=2.0), dict(z=3.0)),
            Points([1.0, 2.0]) + Points([3.0, 3.0])
        )

    def test_uncollate_layout_kdims_and_streams(self):
        layout = self.dmap_ab + self.dmap_xy
        uncollated = layout.uncollate()
        self.assertIsInstance(uncollated, DynamicMap)
        self.assertEqual(uncollated.kdims, self.dmap_ab.kdims)

        # Update streams
        uncollated.streams[0].event(x=3.0, y=4.0)
        self.assertEqual(
            uncollated[1.0, 2.0],
            Points([1.0, 2.0]) + Points([3.0, 4.0])
        )

        self.assertEqual(
            uncollated.callback.callable(1.0, 2.0, dict(x=3.0, y=4.0)),
            Points([1.0, 2.0]) + Points([3.0, 4.0])
        )

    def test_uncollate_spread(self):
        uncollated = self.dmap_spread_points.uncollate()
        self.assertIsInstance(uncollated, DynamicMap)

        # Check top-level stream types
        self.assertEqual(
            [PlotSize, RangeXY, PX],
            [type(s) for s in uncollated.streams]
        )

        # Get expected
        self.px_stream.event(px=3)
        plot_size, range_xy = self.dmap_spread_points.callback.inputs[0].streams
        plot_size.event(width=250, height=300)
        range_xy.event(x_range=(0, 10), y_range=(0, 15))
        expected = self.dmap_spread_points[()]

        # Call uncollated callback function
        result = uncollated.callback.callable(
            {"width": 250, "height": 300},
            {"x_range": (0, 10), "y_range": (0, 15)},
            {"px": 3}
        )

        self.assertEqual(expected, result)

    def test_uncollate_datashade_kdims(self):
        uncollated = self.dmap_datashade_kdim_points.uncollate()
        self.assertIsInstance(uncollated, DynamicMap)

        # Check kdims
        self.assertEqual(uncollated.kdims, self.dmap_ab.kdims)

        # Check top-level stream types
        self.assertEqual(
            [PlotSize, RangeXY],
            [type(s) for s in uncollated.streams]
        )

        # Get expected
        self.px_stream.event(px=3)
        plot_size, range_xy = self.dmap_datashade_kdim_points.streams
        plot_size.event(width=250, height=300)
        range_xy.event(x_range=(0, 10), y_range=(0, 15))
        expected = self.dmap_datashade_kdim_points[4.0, 5.0]

        # Call uncollated callback function
        result = uncollated.callback.callable(
            4.0, 5.0,
            {"width": 250, "height": 300},
            {"x_range": (0, 10), "y_range": (0, 15)},
        )

        self.assertEqual(expected, result)

    def test_uncollate_datashade_kdims_layout(self):
        layout = self.dmap_datashade_kdim_points + self.dmap_b

        uncollated = layout.uncollate()
        self.assertIsInstance(uncollated, DynamicMap)

        # Check kdims
        self.assertEqual(uncollated.kdims, self.dmap_ab.kdims)

        # Check top-level stream types
        self.assertEqual(
            [PlotSize, RangeXY],
            [type(s) for s in uncollated.streams]
        )

        # Get expected
        plot_size, range_xy = self.dmap_datashade_kdim_points.streams
        plot_size.event(width=250, height=300)
        range_xy.event(x_range=(0, 10), y_range=(0, 15))
        expected = self.dmap_datashade_kdim_points[4.0, 5.0] + self.dmap_b[5.0]

        # Call uncollated callback function
        result = uncollated.callback.callable(
            4.0, 5.0,
            {"width": 250, "height": 300},
            {"x_range": (0, 10), "y_range": (0, 15)},
        )

        self.assertEqual(expected, result)

    def test_uncollate_overlay_of_dmaps(self):
        overlay = Overlay([
            DynamicMap(lambda z: Points([z, z]), streams=[Z()]),
            DynamicMap(lambda z: Points([z, z]), streams=[Z()]),
            DynamicMap(lambda z: Points([z, z]), streams=[Z()]),
        ])

        uncollated = overlay.uncollate()
        self.assertIsInstance(uncollated, DynamicMap)
        self.assertEqual(len(uncollated.streams), 3)

        expected = Overlay([
            Points([1.0, 1.0]), Points([2.0, 2.0]), Points([3.0, 3.0])
        ])

        # Build result by updating streams
        uncollated.streams[0].event(z=1.0)
        uncollated.streams[1].event(z=2.0)
        uncollated.streams[2].event(z=3.0)
        result = uncollated[()]
        self.assertEqual(expected, result)

        # Build result by calling callback function
        result = uncollated.callback.callable(dict(z=1.0), dict(z=2.0), dict(z=3.0))
        self.assertEqual(expected, result)


    def test_uncollate_dmap_gridspace_kdims(self):
        self.perform_uncollate_dmap_container_kdims(GridSpace)

    def test_uncollate_dmap_ndoverlay_kdims(self):
        self.perform_uncollate_dmap_container_kdims(NdOverlay)

    def test_uncollate_dmap_holomap_kdims(self):
        self.perform_uncollate_dmap_container_kdims(HoloMap)

    def perform_uncollate_dmap_container_kdims(self, ContainerType):
        # Create container of DynamicMaps, each with kdims a and b.
        data = [
            (0, self.dmap_ab.clone()),
            (1, self.dmap_ab.clone()),
            (2, self.dmap_ab.clone())
        ]
        container = ContainerType(data, kdims=["c"])

        # Uncollate container
        uncollated = container.uncollate()
        self.assertIsInstance(uncollated, DynamicMap)
        self.assertEqual(uncollated.kdims, self.dmap_ab.kdims)

        # Check result of instantiating uncollate DynamicMap for particular kdim values
        a, b = 2.0, 3.0
        expected_data = [(d[0], d[1][a, b]) for d in data]
        expected = ContainerType(expected_data, kdims=["c"])
        result = uncollated[a, b]
        self.assertEqual(expected, result)

    def test_uncollate_dmap_gridspace_streams(self):
        self.perform_uncollate_dmap_container_streams(GridSpace)

    def test_uncollate_dmap_ndoverlay_streams(self):
        self.perform_uncollate_dmap_container_streams(NdOverlay)

    def test_uncollate_dmap_holomap_streams(self):
        self.perform_uncollate_dmap_container_streams(HoloMap)

    def perform_uncollate_dmap_container_streams(self, ContainerType):
        # Create container of DynamicMaps, each with kdims a and b.
        xy_stream = XY()
        fn = lambda x, y: Points([x, y])
        data = [
            (0, DynamicMap(fn, streams=[xy_stream])),
            (1, DynamicMap(fn, streams=[xy_stream])),
            (2, DynamicMap(fn, streams=[xy_stream]))
        ]
        container = ContainerType(data, kdims=["c"])

        # Uncollate container
        uncollated = container.uncollate()
        self.assertIsInstance(uncollated, DynamicMap)
        self.assertEqual(len(uncollated.kdims), 0)
        self.assertEqual(len(uncollated.streams), 1)

        # Check result of instantiating uncollate DynamicMap for particular
        # stream values
        uncollated.streams[0].event(x=2.0, y=3.0)
        xy_stream.event(x=2.0, y=3.0)
        expected_data = [(d[0], d[1][()]) for d in data]
        expected = ContainerType(expected_data, kdims=["c"])
        result = uncollated[()]
        self.assertEqual(expected, result)

    def test_traverse_derived_streams(self):
        from holoviews.tests.teststreams import Val
        uncollated = self.dmap_derived.uncollate()

        # Check uncollated types
        self.assertIsInstance(uncollated, DynamicMap)
        self.assertEqual(len(uncollated.streams), 3)
        for stream in uncollated.streams:
            self.assertIsInstance(stream, Val)

        # Compute expected result
        expected = self.dmap_derived.callback.callable(6.0)
        uncollated.streams[0].event(v=1.0)
        uncollated.streams[1].event(v=2.0)
        uncollated.streams[2].event(v=3.0)
        result = uncollated[()]

        self.assertEqual(expected, result)
