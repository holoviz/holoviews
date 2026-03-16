import numpy as np
import pandas as pd
import param

import holoviews as hv
from holoviews.element.path import BaseShape
from holoviews.testing import assert_data_equal, assert_element_equal


class ElementConstructorTest:
    """
    Tests allowable data formats when constructing
    the basic Element types.
    """

    def setup_method(self):
        self.xs = np.linspace(0, 2 * np.pi, 11)
        self.hxs = np.arange(len(self.xs))
        self.sin = np.sin(self.xs)
        self.cos = np.cos(self.xs)
        sine_data = np.column_stack((self.xs, self.sin))
        cos_data = np.column_stack((self.xs, self.cos))
        self.curve = hv.Curve(sine_data)
        self.path = hv.Path([sine_data, cos_data])
        self.histogram = hv.Histogram((self.hxs, self.sin))

    def test_empty_element_constructor(self):
        failed_elements = []
        for name, el in param.concrete_descendents(hv.Element).items():
            if name == "Sankey":
                continue
            if issubclass(el, (hv.Annotation, BaseShape, hv.Div, hv.Tiles)):
                continue
            try:
                el([])
            except Exception:
                failed_elements.append(name)
        assert failed_elements == []

    def test_chart_zipconstruct(self):
        assert_element_equal(hv.Curve(zip(self.xs, self.sin, strict=True)), self.curve)

    def test_chart_tuple_construct(self):
        assert_element_equal(hv.Curve((self.xs, self.sin)), self.curve)

    def test_path_tuple_construct(self):
        assert_element_equal(hv.Path((self.xs, np.column_stack((self.sin, self.cos)))), self.path)

    def test_path_tuplelist_construct(self):
        assert_element_equal(hv.Path([(self.xs, self.sin), (self.xs, self.cos)]), self.path)

    def test_path_ziplist_construct(self):
        assert_element_equal(
            hv.Path(
                [
                    list(zip(self.xs, self.sin, strict=True)),
                    list(zip(self.xs, self.cos, strict=True)),
                ]
            ),
            self.path,
        )

    def test_hist_zip_construct(self):
        assert_element_equal(
            hv.Histogram(list(zip(self.hxs, self.sin, strict=True))), self.histogram
        )

    def test_hist_array_construct(self):
        assert_element_equal(hv.Histogram(np.column_stack((self.hxs, self.sin))), self.histogram)

    def test_hist_yvalues_construct(self):
        assert_element_equal(hv.Histogram(self.sin), self.histogram)

    def test_hist_curve_construct(self):
        hist = hv.Histogram(hv.Curve(([0.1, 0.3, 0.5], [2.1, 2.2, 3.3])))
        values = hist.dimension_values(1)
        edges = hist.edges
        assert_data_equal(values, np.array([2.1, 2.2, 3.3]))
        assert_data_equal(edges, np.array([0, 0.2, 0.4, 0.6]))

    def test_hist_curve_int_edges_construct(self):
        hist = hv.Histogram(hv.Curve(range(3)))
        values = hist.dimension_values(1)
        edges = hist.edges
        assert_data_equal(values, np.arange(3))
        assert_data_equal(edges, np.array([-0.5, 0.5, 1.5, 2.5]))

    def test_heatmap_construct(self):
        hmap = hv.HeatMap([("A", "a", 1), ("B", "b", 2)])
        dataset = hv.Dataset(
            {"x": ["A", "B"], "y": ["a", "b"], "z": [[1, np.nan], [np.nan, 2]]},
            kdims=["x", "y"],
            vdims=["z"],
            label="unique",
        )
        assert_element_equal(hmap.gridded, dataset)

    def test_heatmap_construct_unsorted(self):
        hmap = hv.HeatMap([("B", "b", 2), ("A", "a", 1)])
        dataset = hv.Dataset(
            {"x": ["B", "A"], "y": ["b", "a"], "z": [[2, np.nan], [np.nan, 1]]},
            kdims=["x", "y"],
            vdims=["z"],
            label="unique",
        )
        assert_element_equal(hmap.gridded, dataset)

    def test_heatmap_construct_partial_sorted(self):
        data = [(chr(65 + i), chr(97 + j), i * j) for i in range(3) for j in [2, 0, 1] if i != j]
        hmap = hv.HeatMap(data)
        dataset = hv.Dataset(
            {
                "x": ["A", "B", "C"],
                "y": ["b", "a", "c"],
                "z": [[0, np.nan, 2], [np.nan, 0, 0], [0, 2, np.nan]],
            },
            kdims=["x", "y"],
            vdims=["z"],
            label="unique",
        )
        assert_element_equal(hmap.gridded, dataset)

    def test_heatmap_construct_and_sort(self):
        data = [(chr(65 + i), chr(97 + j), i * j) for i in range(3) for j in [2, 0, 1] if i != j]
        hmap = hv.HeatMap(data).sort()
        dataset = hv.Dataset(
            {
                "x": ["A", "B", "C"],
                "y": ["a", "b", "c"],
                "z": [[np.nan, 0, 0], [0, np.nan, 2], [0, 2, np.nan]],
            },
            kdims=["x", "y"],
            vdims=["z"],
            label="unique",
        )
        assert_element_equal(hmap.gridded, dataset)


class ElementSignatureTest:
    """
    Test that Element signatures are consistent.
    """

    def test_curve_string_signature(self):
        curve = hv.Curve([], "a", "b")
        assert curve.kdims == [hv.Dimension("a")]
        assert curve.vdims == [hv.Dimension("b")]

    def test_area_string_signature(self):
        area = hv.Area([], "a", "b")
        assert area.kdims == [hv.Dimension("a")]
        assert area.vdims == [hv.Dimension("b")]

    def test_errorbars_string_signature(self):
        errorbars = hv.ErrorBars([], "a", ["b", "c"])
        assert errorbars.kdims == [hv.Dimension("a")]
        assert errorbars.vdims == [hv.Dimension("b"), hv.Dimension("c")]

    def test_bars_string_signature(self):
        bars = hv.Bars([], "a", "b")
        assert bars.kdims == [hv.Dimension("a")]
        assert bars.vdims == [hv.Dimension("b")]

    def test_boxwhisker_string_signature(self):
        boxwhisker = hv.BoxWhisker([], "a", "b")
        assert boxwhisker.kdims == [hv.Dimension("a")]
        assert boxwhisker.vdims == [hv.Dimension("b")]

    def test_scatter_string_signature(self):
        scatter = hv.Scatter([], "a", "b")
        assert scatter.kdims == [hv.Dimension("a")]
        assert scatter.vdims == [hv.Dimension("b")]

    def test_points_string_signature(self):
        points = hv.Points([], ["a", "b"], "c")
        assert points.kdims == [hv.Dimension("a"), hv.Dimension("b")]
        assert points.vdims == [hv.Dimension("c")]

    def test_vectorfield_string_signature(self):
        vectorfield = hv.VectorField([], ["a", "b"], ["c", "d"])
        assert vectorfield.kdims == [hv.Dimension("a"), hv.Dimension("b")]
        assert vectorfield.vdims == [hv.Dimension("c"), hv.Dimension("d")]

    def test_vectorfield_from_uv(self):
        x = np.linspace(-1, 1, 4)
        X, Y = np.meshgrid(x, x)
        U, V = 3 * X, 4 * Y
        vectorfield = hv.VectorField.from_uv((X, Y, U, V))

        angle = np.arctan2(V, U)
        mag = np.hypot(U, V)
        kdims = [hv.Dimension("x"), hv.Dimension("y")]
        vdims = [
            hv.Dimension("Angle", cyclic=True, range=(0, 2 * np.pi)),
            hv.Dimension("Magnitude"),
        ]
        assert vectorfield.kdims == kdims
        assert vectorfield.vdims == vdims
        assert_data_equal(vectorfield.dimension_values(0), X.T.flatten())
        assert_data_equal(vectorfield.dimension_values(1), Y.T.flatten())
        assert_data_equal(vectorfield.dimension_values(2), angle.T.flatten())
        assert_data_equal(vectorfield.dimension_values(3), mag.T.flatten())

    def test_vectorfield_from_uv_dataframe(self):
        x = np.linspace(-1, 1, 4)
        X, Y = np.meshgrid(x, x)
        U, V = 5 * X, 5 * Y
        df = pd.DataFrame(
            {
                "x": X.flatten(),
                "y": Y.flatten(),
                "u": U.flatten(),
                "v": V.flatten(),
            }
        )
        vectorfield = hv.VectorField.from_uv(df, ["x", "y"], ["u", "v"])

        angle = np.arctan2(V, U)
        mag = np.hypot(U, V)
        kdims = [hv.Dimension("x"), hv.Dimension("y")]
        vdims = [
            hv.Dimension("Angle", cyclic=True, range=(0, 2 * np.pi)),
            hv.Dimension("Magnitude"),
        ]
        assert vectorfield.kdims == kdims
        assert vectorfield.vdims == vdims
        np.testing.assert_equal(vectorfield.dimension_values(2, flat=False), angle.flat)
        np.testing.assert_equal(vectorfield.dimension_values(3, flat=False), mag.flat)

    def test_path_string_signature(self):
        path = hv.Path([], ["a", "b"])
        assert path.kdims == [hv.Dimension("a"), hv.Dimension("b")]

    def test_spikes_string_signature(self):
        spikes = hv.Spikes([], "a")
        assert spikes.kdims == [hv.Dimension("a")]

    def test_contours_string_signature(self):
        contours = hv.Contours([], ["a", "b"])
        assert contours.kdims == [hv.Dimension("a"), hv.Dimension("b")]

    def test_polygons_string_signature(self):
        polygons = hv.Polygons([], ["a", "b"])
        assert polygons.kdims == [hv.Dimension("a"), hv.Dimension("b")]

    def test_heatmap_string_signature(self):
        heatmap = hv.HeatMap([], ["a", "b"], "c")
        assert heatmap.kdims == [hv.Dimension("a"), hv.Dimension("b")]
        assert heatmap.vdims == [hv.Dimension("c")]

    def test_raster_string_signature(self):
        raster = hv.Raster(np.array([[0]]), ["a", "b"], "c")
        assert raster.kdims == [hv.Dimension("a"), hv.Dimension("b")]
        assert raster.vdims == [hv.Dimension("c")]

    def test_image_string_signature(self):
        img = hv.Image(np.array([[0, 1], [0, 1]]), ["a", "b"], "c")
        assert img.kdims == [hv.Dimension("a"), hv.Dimension("b")]
        assert img.vdims == [hv.Dimension("c")]

    def test_rgb_string_signature(self):
        img = hv.RGB(np.zeros((2, 2, 3)), ["a", "b"], ["R", "G", "B"])
        assert img.kdims == [hv.Dimension("a"), hv.Dimension("b")]
        assert img.vdims == [hv.Dimension("R"), hv.Dimension("G"), hv.Dimension("B")]

    def test_quadmesh_string_signature(self):
        qmesh = hv.QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [0, 1]])), ["a", "b"], "c")
        assert qmesh.kdims == [hv.Dimension("a"), hv.Dimension("b")]
        assert qmesh.vdims == [hv.Dimension("c")]


class ElementCastingTests:
    """
    Tests whether casting an element will faithfully copy data and
    parameters. Important to check for elements where data is not all
    held on .data attribute, e.g. Image bounds or Graph nodes and
    edgepaths.
    """

    def test_image_casting(self):
        img = hv.Image([], bounds=2)
        assert_element_equal(img, hv.Image(img))

    def test_rgb_casting(self):
        rgb = hv.RGB([], bounds=2)
        assert_element_equal(rgb, hv.RGB(rgb))

    def test_graph_casting(self):
        graph = hv.Graph(([(0, 1)], [(0, 0, 0), (0, 1, 1)]))
        assert_element_equal(graph, hv.Graph(graph))

    def test_trimesh_casting(self):
        trimesh = hv.TriMesh(([(0, 1, 2)], [(0, 0, 0), (0, 1, 1), (1, 1, 2)]))
        assert_element_equal(trimesh, hv.TriMesh(trimesh))
