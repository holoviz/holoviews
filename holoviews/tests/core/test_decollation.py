from unittest import skipIf

import param

from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream

try:
    from holoviews.operation.datashader import datashade, spread
except ImportError:
    spread = datashade = None

datashade_skip = skipIf(datashade is None, "datashade is not available")


class XY(Stream):
    x = param.Number(constant=True)
    y = param.Number(constant=True)


Z = Stream.define("Z", z=0.0)

PX = Stream.define("PX", px=1)


class TestDecollation(ComparisonTestCase):
    def setUp(self):
        from holoviews.tests.test_streams import Sum, Val

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

        if datashade is None:
            return

        # dmap produced by chained datashade and shade
        self.px_stream = PX()
        self.dmap_spread_points = spread(
            datashade(Points([0.0, 1.0])), streams=[self.px_stream]
        )

        # data shaded with kdims: a, b
        self.dmap_datashade_kdim_points = datashade(self.dmap_ab)


    def test_decollate_layout_kdims(self):
        layout = self.dmap_ab + self.dmap_b
        decollated = layout.decollate()
        self.assertIsInstance(decollated, DynamicMap)
        self.assertEqual(decollated.kdims, self.dmap_ab.kdims)
        self.assertEqual(
            decollated[2, 3],
            Points([2, 3]) + Points([3, 3])
        )
        self.assertEqual(
            decollated.callback.callable(2, 3),
            Points([2, 3]) + Points([3, 3])
        )

    def test_decollate_layout_streams(self):
        layout = self.dmap_xy + self.dmap_z
        decollated = layout.decollate()
        self.assertIsInstance(decollated, DynamicMap)
        self.assertEqual(decollated.kdims, [])

        # Update streams
        decollated.streams[0].event(x=1.0, y=2.0)
        decollated.streams[1].event(z=3.0)
        self.assertEqual(
            decollated[()],
            Points([1.0, 2.0]) + Points([3.0, 3.0])
        )
        self.assertEqual(
            decollated.callback.callable(dict(x=1.0, y=2.0), dict(z=3.0)),
            Points([1.0, 2.0]) + Points([3.0, 3.0])
        )

    def test_decollate_layout_kdims_and_streams(self):
        layout = self.dmap_ab + self.dmap_xy
        decollated = layout.decollate()
        self.assertIsInstance(decollated, DynamicMap)
        self.assertEqual(decollated.kdims, self.dmap_ab.kdims)

        # Update streams
        decollated.streams[0].event(x=3.0, y=4.0)
        self.assertEqual(
            decollated[1.0, 2.0],
            Points([1.0, 2.0]) + Points([3.0, 4.0])
        )

        self.assertEqual(
            decollated.callback.callable(1.0, 2.0, dict(x=3.0, y=4.0)),
            Points([1.0, 2.0]) + Points([3.0, 4.0])
        )

    @datashade_skip
    def test_decollate_spread(self):
        decollated = self.dmap_spread_points.decollate()
        self.assertIsInstance(decollated, DynamicMap)

        # Check top-level stream types
        self.assertEqual(
            [PlotSize, RangeXY, PX],
            [type(s) for s in decollated.streams]
        )

        # Get expected
        self.px_stream.event(px=3)
        plot_size, range_xy = self.dmap_spread_points.callback.inputs[0].streams
        plot_size.event(width=250, height=300)
        range_xy.event(x_range=(0, 10), y_range=(0, 15))
        expected = self.dmap_spread_points[()]

        # Call decollated callback function
        result = decollated.callback.callable(
            {"width": 250, "height": 300},
            {"x_range": (0, 10), "y_range": (0, 15)},
            {"px": 3}
        )

        self.assertEqual(expected, result)

    @datashade_skip
    def test_decollate_datashade_kdims(self):
        decollated = self.dmap_datashade_kdim_points.decollate()
        self.assertIsInstance(decollated, DynamicMap)

        # Check kdims
        self.assertEqual(decollated.kdims, self.dmap_ab.kdims)

        # Check top-level stream types
        self.assertEqual(
            [PlotSize, RangeXY],
            [type(s) for s in decollated.streams]
        )

        # Get expected
        self.px_stream.event(px=3)
        plot_size, range_xy = self.dmap_datashade_kdim_points.streams
        plot_size.event(width=250, height=300)
        range_xy.event(x_range=(0, 10), y_range=(0, 15))
        expected = self.dmap_datashade_kdim_points[4.0, 5.0]

        # Call decollated callback function
        result = decollated.callback.callable(
            4.0, 5.0,
            {"width": 250, "height": 300},
            {"x_range": (0, 10), "y_range": (0, 15)},
        )

        self.assertEqual(expected, result)


    @datashade_skip
    def test_decollate_datashade_kdims_layout(self):
        layout = self.dmap_datashade_kdim_points + self.dmap_b

        decollated = layout.decollate()
        self.assertIsInstance(decollated, DynamicMap)

        # Check kdims
        self.assertEqual(decollated.kdims, self.dmap_ab.kdims)

        # Check top-level stream types
        self.assertEqual(
            [PlotSize, RangeXY],
            [type(s) for s in decollated.streams]
        )

        # Get expected
        plot_size, range_xy = self.dmap_datashade_kdim_points.streams
        plot_size.event(width=250, height=300)
        range_xy.event(x_range=(0, 10), y_range=(0, 15))
        expected = self.dmap_datashade_kdim_points[4.0, 5.0] + self.dmap_b[5.0]

        # Call decollated callback function
        result = decollated.callback.callable(
            4.0, 5.0,
            {"width": 250, "height": 300},
            {"x_range": (0, 10), "y_range": (0, 15)},
        )

        self.assertEqual(expected, result)

    def test_decollate_overlay_of_dmaps(self):
        overlay = Overlay([
            DynamicMap(lambda z: Points([z, z]), streams=[Z()]),
            DynamicMap(lambda z: Points([z, z]), streams=[Z()]),
            DynamicMap(lambda z: Points([z, z]), streams=[Z()]),
        ])

        decollated = overlay.decollate()
        self.assertIsInstance(decollated, DynamicMap)
        self.assertEqual(len(decollated.streams), 3)

        expected = Overlay([
            Points([1.0, 1.0]), Points([2.0, 2.0]), Points([3.0, 3.0])
        ])

        # Build result by updating streams
        decollated.streams[0].event(z=1.0)
        decollated.streams[1].event(z=2.0)
        decollated.streams[2].event(z=3.0)
        result = decollated[()]
        self.assertEqual(expected, result)

        # Build result by calling callback function
        result = decollated.callback.callable(dict(z=1.0), dict(z=2.0), dict(z=3.0))
        self.assertEqual(expected, result)


    def test_decollate_dmap_gridspace_kdims(self):
        self.perform_decollate_dmap_container_kdims(GridSpace)

    def test_decollate_dmap_ndoverlay_kdims(self):
        self.perform_decollate_dmap_container_kdims(NdOverlay)

    def test_decollate_dmap_holomap_kdims(self):
        self.perform_decollate_dmap_container_kdims(HoloMap)

    def perform_decollate_dmap_container_kdims(self, ContainerType):
        # Create container of DynamicMaps, each with kdims a and b.
        data = [
            (0, self.dmap_ab.clone()),
            (1, self.dmap_ab.clone()),
            (2, self.dmap_ab.clone())
        ]
        container = ContainerType(data, kdims=["c"])

        # Decollate container
        decollated = container.decollate()
        self.assertIsInstance(decollated, DynamicMap)
        self.assertEqual(decollated.kdims, self.dmap_ab.kdims)

        # Check result of instantiating decollate DynamicMap for particular kdim values
        a, b = 2.0, 3.0
        expected_data = [(d[0], d[1][a, b]) for d in data]
        expected = ContainerType(expected_data, kdims=["c"])
        result = decollated[a, b]
        self.assertEqual(expected, result)

    def test_decollate_dmap_gridspace_streams(self):
        self.perform_decollate_dmap_container_streams(GridSpace)

    def test_decollate_dmap_ndoverlay_streams(self):
        self.perform_decollate_dmap_container_streams(NdOverlay)

    def test_decollate_dmap_holomap_streams(self):
        self.perform_decollate_dmap_container_streams(HoloMap)

    def perform_decollate_dmap_container_streams(self, ContainerType):
        # Create container of DynamicMaps, each with kdims a and b.
        xy_stream = XY()
        fn = lambda x, y: Points([x, y])
        data = [
            (0, DynamicMap(fn, streams=[xy_stream])),
            (1, DynamicMap(fn, streams=[xy_stream])),
            (2, DynamicMap(fn, streams=[xy_stream]))
        ]
        container = ContainerType(data, kdims=["c"])

        # Decollate container
        decollated = container.decollate()
        self.assertIsInstance(decollated, DynamicMap)
        self.assertEqual(len(decollated.kdims), 0)
        self.assertEqual(len(decollated.streams), 1)

        # Check result of instantiating decollate DynamicMap for particular
        # stream values
        decollated.streams[0].event(x=2.0, y=3.0)
        xy_stream.event(x=2.0, y=3.0)
        expected_data = [(d[0], d[1][()]) for d in data]
        expected = ContainerType(expected_data, kdims=["c"])
        result = decollated[()]
        self.assertEqual(expected, result)

    def test_traverse_derived_streams(self):
        from holoviews.tests.test_streams import Val
        decollated = self.dmap_derived.decollate()

        # Check decollated types
        self.assertIsInstance(decollated, DynamicMap)
        self.assertEqual(len(decollated.streams), 3)
        for stream in decollated.streams:
            self.assertIsInstance(stream, Val)

        # Compute expected result
        expected = self.dmap_derived.callback.callable(6.0)
        decollated.streams[0].event(v=1.0)
        decollated.streams[1].event(v=2.0)
        decollated.streams[2].event(v=3.0)
        result = decollated[()]

        self.assertEqual(expected, result)
