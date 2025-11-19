import numpy as np
import pandas as pd
import param

from holoviews import (
    RGB,
    Annotation,
    Area,
    Bars,
    BoxWhisker,
    Contours,
    Curve,
    Dataset,
    Dimension,
    Div,
    Element,
    ErrorBars,
    Graph,
    HeatMap,
    Histogram,
    Image,
    Path,
    Points,
    Polygons,
    QuadMesh,
    Raster,
    Scatter,
    Spikes,
    Tiles,
    TriMesh,
    VectorField,
)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
from holoviews.testing import assert_data_equal, assert_element_equal


class ElementConstructorTest(ComparisonTestCase):
    """
    Tests allowable data formats when constructing
    the basic Element types.
    """

    def setUp(self):
        self.xs = np.linspace(0, 2*np.pi, 11)
        self.hxs = np.arange(len(self.xs))
        self.sin = np.sin(self.xs)
        self.cos = np.cos(self.xs)
        sine_data = np.column_stack((self.xs, self.sin))
        cos_data = np.column_stack((self.xs, self.cos))
        self.curve = Curve(sine_data)
        self.path = Path([sine_data, cos_data])
        self.histogram = Histogram((self.hxs, self.sin))
        super().setUp()

    def test_empty_element_constructor(self):
        failed_elements = []
        for name, el in param.concrete_descendents(Element).items():
            if name == 'Sankey': continue
            if issubclass(el, (Annotation, BaseShape, Div, Tiles)):
                continue
            try:
                el([])
            except Exception:
                failed_elements.append(name)
        assert failed_elements == []

    def test_chart_zipconstruct(self):
        assert_element_equal(Curve(zip(self.xs, self.sin, strict=None)), self.curve)

    def test_chart_tuple_construct(self):
        assert_element_equal(Curve((self.xs, self.sin)), self.curve)

    def test_path_tuple_construct(self):
        assert_element_equal(Path((self.xs, np.column_stack((self.sin, self.cos)))), self.path)

    def test_path_tuplelist_construct(self):
        assert_element_equal(Path([(self.xs, self.sin), (self.xs, self.cos)]), self.path)

    def test_path_ziplist_construct(self):
        assert_element_equal(Path([list(zip(self.xs, self.sin, strict=None)), list(zip(self.xs, self.cos, strict=None))]), self.path)

    def test_hist_zip_construct(self):
        assert_element_equal(Histogram(list(zip(self.hxs, self.sin, strict=None))), self.histogram)

    def test_hist_array_construct(self):
        assert_element_equal(Histogram(np.column_stack((self.hxs, self.sin))), self.histogram)

    def test_hist_yvalues_construct(self):
        assert_element_equal(Histogram(self.sin), self.histogram)

    def test_hist_curve_construct(self):
        hist = Histogram(Curve(([0.1, 0.3, 0.5], [2.1, 2.2, 3.3])))
        values = hist.dimension_values(1)
        edges = hist.edges
        assert_data_equal(values, np.array([2.1, 2.2, 3.3]))
        assert_data_equal(edges, np.array([0, 0.2, 0.4, 0.6]))

    def test_hist_curve_int_edges_construct(self):
        hist = Histogram(Curve(range(3)))
        values = hist.dimension_values(1)
        edges = hist.edges
        assert_data_equal(values, np.arange(3))
        assert_data_equal(edges, np.array([-.5, .5, 1.5, 2.5]))

    def test_heatmap_construct(self):
        hmap = HeatMap([('A', 'a', 1), ('B', 'b', 2)])
        dataset = Dataset({'x': ['A', 'B'], 'y': ['a', 'b'], 'z': [[1, np.nan], [np.nan, 2]]},
                          kdims=['x', 'y'], vdims=['z'], label='unique')
        assert_element_equal(hmap.gridded, dataset)

    def test_heatmap_construct_unsorted(self):
        hmap = HeatMap([('B', 'b', 2), ('A', 'a', 1)])
        dataset = Dataset({'x': ['B', 'A'], 'y': ['b', 'a'], 'z': [[2, np.nan], [np.nan, 1]]},
                          kdims=['x', 'y'], vdims=['z'], label='unique')
        assert_element_equal(hmap.gridded, dataset)

    def test_heatmap_construct_partial_sorted(self):
        data = [(chr(65+i),chr(97+j), i*j) for i in range(3) for j in [2, 0, 1] if i!=j]
        hmap = HeatMap(data)
        dataset = Dataset({
            'x': ['A', 'B', 'C'],
            'y': ['b', 'a', 'c'],
            'z': [[0, np.nan, 2], [np.nan, 0, 0], [0, 2, np.nan]]
        }, kdims=['x', 'y'], vdims=['z'], label='unique')
        assert_element_equal(hmap.gridded, dataset)

    def test_heatmap_construct_and_sort(self):
        data = [(chr(65+i),chr(97+j), i*j) for i in range(3) for j in [2, 0, 1] if i!=j]
        hmap = HeatMap(data).sort()
        dataset = Dataset({'x': ['A', 'B', 'C'], 'y': ['a', 'b', 'c'],
                           'z': [[np.nan, 0, 0], [0, np.nan, 2], [0, 2, np.nan]]},
                          kdims=['x', 'y'], vdims=['z'], label='unique')
        assert_element_equal(hmap.gridded, dataset)



class ElementSignatureTest(ComparisonTestCase):
    """
    Test that Element signatures are consistent.
    """

    def test_curve_string_signature(self):
        curve = Curve([], 'a', 'b')
        assert curve.kdims == [Dimension('a')]
        assert curve.vdims == [Dimension('b')]

    def test_area_string_signature(self):
        area = Area([], 'a', 'b')
        assert area.kdims == [Dimension('a')]
        assert area.vdims == [Dimension('b')]

    def test_errorbars_string_signature(self):
        errorbars = ErrorBars([], 'a', ['b', 'c'])
        assert errorbars.kdims == [Dimension('a')]
        assert errorbars.vdims == [Dimension('b'), Dimension('c')]

    def test_bars_string_signature(self):
        bars = Bars([], 'a', 'b')
        assert bars.kdims == [Dimension('a')]
        assert bars.vdims == [Dimension('b')]

    def test_boxwhisker_string_signature(self):
        boxwhisker = BoxWhisker([], 'a', 'b')
        assert boxwhisker.kdims == [Dimension('a')]
        assert boxwhisker.vdims == [Dimension('b')]

    def test_scatter_string_signature(self):
        scatter = Scatter([], 'a', 'b')
        assert scatter.kdims == [Dimension('a')]
        assert scatter.vdims == [Dimension('b')]

    def test_points_string_signature(self):
        points = Points([], ['a', 'b'], 'c')
        assert points.kdims == [Dimension('a'), Dimension('b')]
        assert points.vdims == [Dimension('c')]

    def test_vectorfield_string_signature(self):
        vectorfield = VectorField([], ['a', 'b'], ['c', 'd'])
        assert vectorfield.kdims == [Dimension('a'), Dimension('b')]
        assert vectorfield.vdims == [Dimension('c'), Dimension('d')]

    def test_vectorfield_from_uv(self):
        x = np.linspace(-1, 1, 4)
        X, Y = np.meshgrid(x, x)
        U, V = 3 * X, 4 * Y
        vectorfield = VectorField.from_uv((X, Y, U, V))

        angle = np.arctan2(V, U)
        mag = np.hypot(U, V)
        kdims = [Dimension('x'), Dimension('y')]
        vdims = [
            Dimension('Angle', cyclic=True, range=(0,2*np.pi)),
            Dimension('Magnitude')
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
        df = pd.DataFrame({
            "x": X.flatten(),
            "y": Y.flatten(),
            "u": U.flatten(),
            "v": V.flatten(),
        })
        vectorfield = VectorField.from_uv(df, ["x", "y"], ["u", "v"])

        angle = np.arctan2(V, U)
        mag = np.hypot(U, V)
        kdims = [Dimension('x'), Dimension('y')]
        vdims = [
            Dimension('Angle', cyclic=True, range=(0,2*np.pi)),
            Dimension('Magnitude')
        ]
        assert vectorfield.kdims == kdims
        assert vectorfield.vdims == vdims
        np.testing.assert_equal(vectorfield.dimension_values(2, flat=False), angle.flat)
        np.testing.assert_equal(vectorfield.dimension_values(3, flat=False), mag.flat)

    def test_path_string_signature(self):
        path = Path([], ['a', 'b'])
        assert path.kdims == [Dimension('a'), Dimension('b')]

    def test_spikes_string_signature(self):
        spikes = Spikes([], 'a')
        assert spikes.kdims == [Dimension('a')]

    def test_contours_string_signature(self):
        contours = Contours([], ['a', 'b'])
        assert contours.kdims == [Dimension('a'), Dimension('b')]

    def test_polygons_string_signature(self):
        polygons = Polygons([], ['a', 'b'])
        assert polygons.kdims == [Dimension('a'), Dimension('b')]

    def test_heatmap_string_signature(self):
        heatmap = HeatMap([], ['a', 'b'], 'c')
        assert heatmap.kdims == [Dimension('a'), Dimension('b')]
        assert heatmap.vdims == [Dimension('c')]

    def test_raster_string_signature(self):
        raster = Raster(np.array([[0]]), ['a', 'b'], 'c')
        assert raster.kdims == [Dimension('a'), Dimension('b')]
        assert raster.vdims == [Dimension('c')]

    def test_image_string_signature(self):
        img = Image(np.array([[0, 1], [0, 1]]), ['a', 'b'], 'c')
        assert img.kdims == [Dimension('a'), Dimension('b')]
        assert img.vdims == [Dimension('c')]

    def test_rgb_string_signature(self):
        img = RGB(np.zeros((2, 2, 3)), ['a', 'b'], ['R', 'G', 'B'])
        assert img.kdims == [Dimension('a'), Dimension('b')]
        assert img.vdims == [Dimension('R'), Dimension('G'), Dimension('B')]

    def test_quadmesh_string_signature(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [0, 1]])), ['a', 'b'], 'c')
        assert qmesh.kdims == [Dimension('a'), Dimension('b')]
        assert qmesh.vdims == [Dimension('c')]


class ElementCastingTests(ComparisonTestCase):
    """
    Tests whether casting an element will faithfully copy data and
    parameters. Important to check for elements where data is not all
    held on .data attribute, e.g. Image bounds or Graph nodes and
    edgepaths.
    """

    def test_image_casting(self):
        img = Image([], bounds=2)
        assert_element_equal(img, Image(img))

    def test_rgb_casting(self):
        rgb = RGB([], bounds=2)
        assert_element_equal(rgb, RGB(rgb))

    def test_graph_casting(self):
        graph = Graph(([(0, 1)], [(0, 0, 0), (0, 1, 1)]))
        assert_element_equal(graph, Graph(graph))

    def test_trimesh_casting(self):
        trimesh = TriMesh(([(0, 1, 2)], [(0, 0, 0), (0, 1, 1), (1, 1, 2)]))
        assert_element_equal(trimesh, TriMesh(trimesh))
