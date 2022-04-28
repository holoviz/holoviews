import param
import numpy as np

from holoviews import (
    Dimension, Dataset, Element, Annotation, Curve, Path, Histogram,
    HeatMap, Contours, Scatter, Points, Polygons, VectorField, Spikes,
    Area, Bars, ErrorBars, BoxWhisker, Raster, Image, QuadMesh, RGB,
    Graph, TriMesh, Div, Tiles
)
from holoviews.element.path import BaseShape
from holoviews.element.comparison import ComparisonTestCase

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
            except:
                failed_elements.append(name)
        self.assertEqual(failed_elements, [])

    def test_chart_zipconstruct(self):
        self.assertEqual(Curve(zip(self.xs, self.sin)), self.curve)

    def test_chart_tuple_construct(self):
        self.assertEqual(Curve((self.xs, self.sin)), self.curve)

    def test_path_tuple_construct(self):
        self.assertEqual(Path((self.xs, np.column_stack((self.sin, self.cos)))), self.path)

    def test_path_tuplelist_construct(self):
        self.assertEqual(Path([(self.xs, self.sin), (self.xs, self.cos)]), self.path)

    def test_path_ziplist_construct(self):
        self.assertEqual(Path([list(zip(self.xs, self.sin)), list(zip(self.xs, self.cos))]), self.path)

    def test_hist_zip_construct(self):
        self.assertEqual(Histogram(list(zip(self.hxs, self.sin))), self.histogram)

    def test_hist_array_construct(self):
        self.assertEqual(Histogram(np.column_stack((self.hxs, self.sin))), self.histogram)

    def test_hist_yvalues_construct(self):
        self.assertEqual(Histogram(self.sin), self.histogram)

    def test_hist_curve_construct(self):
        hist = Histogram(Curve(([0.1, 0.3, 0.5], [2.1, 2.2, 3.3])))
        values = hist.dimension_values(1)
        edges = hist.edges
        self.assertEqual(values, np.array([2.1, 2.2, 3.3]))
        self.assertEqual(edges, np.array([0, 0.2, 0.4, 0.6]))

    def test_hist_curve_int_edges_construct(self):
        hist = Histogram(Curve(range(3)))
        values = hist.dimension_values(1)
        edges = hist.edges
        self.assertEqual(values, np.arange(3))
        self.assertEqual(edges, np.array([-.5, .5, 1.5, 2.5]))

    def test_heatmap_construct(self):
        hmap = HeatMap([('A', 'a', 1), ('B', 'b', 2)])
        dataset = Dataset({'x': ['A', 'B'], 'y': ['a', 'b'], 'z': [[1, np.NaN], [np.NaN, 2]]},
                          kdims=['x', 'y'], vdims=['z'], label='unique')
        self.assertEqual(hmap.gridded, dataset)

    def test_heatmap_construct_unsorted(self):
        hmap = HeatMap([('B', 'b', 2), ('A', 'a', 1)])
        dataset = Dataset({'x': ['B', 'A'], 'y': ['b', 'a'], 'z': [[2, np.NaN], [np.NaN, 1]]},
                          kdims=['x', 'y'], vdims=['z'], label='unique')
        self.assertEqual(hmap.gridded, dataset)

    def test_heatmap_construct_partial_sorted(self):
        data = [(chr(65+i),chr(97+j), i*j) for i in range(3) for j in [2, 0, 1] if i!=j]
        hmap = HeatMap(data)
        dataset = Dataset({'x': ['A', 'B', 'C'], 'y': ['c', 'b', 'a'],
                           'z': [[0, 2, np.NaN], [np.NaN, 0, 0], [0, np.NaN, 2]]},
                          kdims=['x', 'y'], vdims=['z'], label='unique')
        self.assertEqual(hmap.gridded, dataset)

    def test_heatmap_construct_and_sort(self):
        data = [(chr(65+i),chr(97+j), i*j) for i in range(3) for j in [2, 0, 1] if i!=j]
        hmap = HeatMap(data).sort()
        dataset = Dataset({'x': ['A', 'B', 'C'], 'y': ['a', 'b', 'c'],
                           'z': [[np.NaN, 0, 0], [0, np.NaN, 2], [0, 2, np.NaN]]},
                          kdims=['x', 'y'], vdims=['z'], label='unique')
        self.assertEqual(hmap.gridded, dataset)



class ElementSignatureTest(ComparisonTestCase):
    """
    Test that Element signatures are consistent.
    """

    def test_curve_string_signature(self):
        curve = Curve([], 'a', 'b')
        self.assertEqual(curve.kdims, [Dimension('a')])
        self.assertEqual(curve.vdims, [Dimension('b')])

    def test_area_string_signature(self):
        area = Area([], 'a', 'b')
        self.assertEqual(area.kdims, [Dimension('a')])
        self.assertEqual(area.vdims, [Dimension('b')])

    def test_errorbars_string_signature(self):
        errorbars = ErrorBars([], 'a', ['b', 'c'])
        self.assertEqual(errorbars.kdims, [Dimension('a')])
        self.assertEqual(errorbars.vdims, [Dimension('b'), Dimension('c')])

    def test_bars_string_signature(self):
        bars = Bars([], 'a', 'b')
        self.assertEqual(bars.kdims, [Dimension('a')])
        self.assertEqual(bars.vdims, [Dimension('b')])

    def test_boxwhisker_string_signature(self):
        boxwhisker = BoxWhisker([], 'a', 'b')
        self.assertEqual(boxwhisker.kdims, [Dimension('a')])
        self.assertEqual(boxwhisker.vdims, [Dimension('b')])

    def test_scatter_string_signature(self):
        scatter = Scatter([], 'a', 'b')
        self.assertEqual(scatter.kdims, [Dimension('a')])
        self.assertEqual(scatter.vdims, [Dimension('b')])

    def test_points_string_signature(self):
        points = Points([], ['a', 'b'], 'c')
        self.assertEqual(points.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(points.vdims, [Dimension('c')])

    def test_vectorfield_string_signature(self):
        vectorfield = VectorField([], ['a', 'b'], ['c', 'd'])
        self.assertEqual(vectorfield.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(vectorfield.vdims, [Dimension('c'), Dimension('d')])

    def test_path_string_signature(self):
        path = Path([], ['a', 'b'])
        self.assertEqual(path.kdims, [Dimension('a'), Dimension('b')])

    def test_spikes_string_signature(self):
        spikes = Spikes([], 'a')
        self.assertEqual(spikes.kdims, [Dimension('a')])

    def test_contours_string_signature(self):
        contours = Contours([], ['a', 'b'])
        self.assertEqual(contours.kdims, [Dimension('a'), Dimension('b')])

    def test_polygons_string_signature(self):
        polygons = Polygons([], ['a', 'b'])
        self.assertEqual(polygons.kdims, [Dimension('a'), Dimension('b')])

    def test_heatmap_string_signature(self):
        heatmap = HeatMap([], ['a', 'b'], 'c')
        self.assertEqual(heatmap.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(heatmap.vdims, [Dimension('c')])

    def test_raster_string_signature(self):
        raster = Raster(np.array([[0]]), ['a', 'b'], 'c')
        self.assertEqual(raster.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(raster.vdims, [Dimension('c')])

    def test_image_string_signature(self):
        img = Image(np.array([[0, 1], [0, 1]]), ['a', 'b'], 'c')
        self.assertEqual(img.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(img.vdims, [Dimension('c')])

    def test_rgb_string_signature(self):
        img = RGB(np.zeros((2, 2, 3)), ['a', 'b'], ['R', 'G', 'B'])
        self.assertEqual(img.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(img.vdims, [Dimension('R'), Dimension('G'), Dimension('B')])

    def test_quadmesh_string_signature(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [0, 1]])), ['a', 'b'], 'c')
        self.assertEqual(qmesh.kdims, [Dimension('a'), Dimension('b')])
        self.assertEqual(qmesh.vdims, [Dimension('c')])


class ElementCastingTests(ComparisonTestCase):
    """
    Tests whether casting an element will faithfully copy data and
    parameters. Important to check for elements where data is not all
    held on .data attribute, e.g. Image bounds or Graph nodes and
    edgepaths.
    """

    def test_image_casting(self):
        img = Image([], bounds=2)
        self.assertEqual(img, Image(img))

    def test_rgb_casting(self):
        rgb = RGB([], bounds=2)
        self.assertEqual(rgb, RGB(rgb))

    def test_graph_casting(self):
        graph = Graph(([(0, 1)], [(0, 0, 0), (0, 1, 1)]))
        self.assertEqual(graph, Graph(graph))

    def test_trimesh_casting(self):
        trimesh = TriMesh(([(0, 1, 2)], [(0, 0, 0), (0, 1, 1), (1, 1, 2)]))
        self.assertEqual(trimesh, TriMesh(trimesh))
