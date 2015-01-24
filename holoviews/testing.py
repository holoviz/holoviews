import unittest
from unittest import SkipTest

from numpy.testing import assert_array_almost_equal
from IPython.display import HTML, SVG

from .core import AdjointLayout, Overlay
from .core.settings import Settings
from .element import *
from holoviews import Matrix
from holoviews.element.annotation import Annotation, Contours
from holoviews.element.raster import Raster
from .interface.pandas import *
from .interface.seaborn import *


class ViewTestCase(unittest.TestCase):
    """
    The class implements comparisons between ViewableElement objects for the
    purposes of testing. The most important attribute that needs to be
    compared is the data attribute as this contains the raw data held
    by the ViewableElement object.
    """
    def __init__(self, *args, **kwargs):
        super(ViewTestCase, self).__init__(*args, **kwargs)
        # General view classes
        self.addTypeEqualityFunc(NdLayout,    self.compare_gridlayout)
        self.addTypeEqualityFunc(AdjointLayout, self.compare_layouts)
        self.addTypeEqualityFunc(NdOverlay,     self.compare_layers)
        self.addTypeEqualityFunc(Overlay,       self.compare_layers)
        self.addTypeEqualityFunc(Annotation,    self.compare_annotations)
        self.addTypeEqualityFunc(AxisLayout,    self.compare_grids)
        self.addTypeEqualityFunc(HoloMap,       self.compare_viewmap)

        # DataLayers
        self.addTypeEqualityFunc(Curve,        self.compare_curve)
        self.addTypeEqualityFunc(Histogram,    self.compare_histogram)
        self.addTypeEqualityFunc(Raster,       self.compare_raster)
        self.addTypeEqualityFunc(HeatMap,      self.compare_heatmap)
        # Tables
        self.addTypeEqualityFunc(ItemTable,    self.compare_itemtables)
        self.addTypeEqualityFunc(Table,        self.compare_tables)
        # SheetLayers
        self.addTypeEqualityFunc(Matrix,       self.compare_matrix)
        self.addTypeEqualityFunc(Contours,     self.compare_contours)
        self.addTypeEqualityFunc(Points,       self.compare_points)
        self.addTypeEqualityFunc(VectorField,  self.compare_vectorfield)

        # Pandas DFrame objects
        self.addTypeEqualityFunc(DataFrameView, self.compare_dframe)
        self.addTypeEqualityFunc(PandasDFrame,  self.compare_dframe)
        self.addTypeEqualityFunc(DFrame,        self.compare_dframe)

        # Seaborn Views
        self.addTypeEqualityFunc(Bivariate,    self.compare_bivariate)
        self.addTypeEqualityFunc(Distribution, self.compare_distribution)
        self.addTypeEqualityFunc(Regression,   self.compare_regression)
        self.addTypeEqualityFunc(TimeSeries,   self.compare_timeseries)

        # Option objects
        self.addTypeEqualityFunc(Settings,     self.compare_settings)
        # Dimension objects
        self.addTypeEqualityFunc(Dimension,    self.compare_dims)

        # Float comparisons
        self.addTypeEqualityFunc(float,        self.compare_floats)
        self.addTypeEqualityFunc(np.float,     self.compare_floats)
        self.addTypeEqualityFunc(np.float32,   self.compare_floats)
        self.addTypeEqualityFunc(np.float64,   self.compare_floats)

    #================#
    # Helper methods #
    #================#

    def compare_floats(self, arr1, arr2, msg='Float'):
        self.compare_arrays(arr1, arr2, msg)

    def compare_arrays(self, arr1, arr2, name):
        try:
            assert_array_almost_equal(arr1, arr2)
        except AssertionError as e:
            raise self.failureException(name+': '+str(e))


    def bounds_check(self, view1, view2):
        if view1.bounds.lbrt() != view2.bounds.lbrt():
            raise self.failureException("BoundingBoxes are mismatched.")


    def compare_maps(self, view1, view2, msg):

        if view1.ndims != view2.ndims:
            raise self.failureException("Maps have different numbers of dimensions.")

        if [d.name for d in view1.dimensions] != [d.name for d in view2.dimensions]:
            raise self.failureException("Maps have different dimension labels.")

        if len(view1.keys()) != len(view2.keys()):
            raise self.failureException("Maps have different numbers of keys.")

        if set(view1.keys()) != set(view2.keys()):
            raise self.failureException("Maps have different sets of keys.")

        for el1, el2 in zip(view1, view2):
            self.assertEqual(el1,el2)

    #=================#
    # Generic classes #
    #=================#

    def compare_viewmap(self, view1, view2, msg):
        self.compare_maps(view1, view2, msg)

    def compare_gridlayout(self, view1, view2, msg):
        if len(view1) != len(view2):
            raise self.failureException("GridLayouts have different sizes.")

        if set(view1.keys()) != set(view2.keys()):
            raise self.failureException("GridLayouts have different keys.")

        for el1, el2 in zip(view1, view2):
            self.assertEqual(el1,el2)

    def compare_layouts(self, view1, view2, msg):
        for el1, el2 in zip(view1, view1):
            self.assertEqual(el1, el2)

    def compare_layers(self, view1, view2, msg):
        if len(view1) != len(view2):
            raise self.failureException("CompositeOverlay have different lengths.")

        for (layer1, layer2) in zip(view1, view2):
            self.assertEqual(layer1, layer2)

    def compare_intervals(self, interval1, interval2):
        if (interval1, interval2) == (None,None):
            return
        elif None in (interval1, interval2):
            raise self.failureException("Mismatched interval annotation types.")
        elif set(interval1.keys()) != set(interval2.keys()):
            raise self.failureException("Mismatched interval annotation keys.")

        for key in interval1.keys():
            (i1s, i1e) = interval1[key]
            (i2s, i2e) = interval2[key]
            if None in [i1s, i2s]:
                self.assertEqual(i1s, i2s)
            else:
                self.compare_floats(i1s, i2s, 'Interval start')
            if None in [i1e, i2e]:
                self.assertEqual(i1e, i2e)
            else:
                self.compare_floats(i1e, i2e, 'Interval end')


    def compare_annotations(self, view1, view2, msg):
        """
        Note: Currently only process vline and hline correctly
        """
        for el1, el2 in zip(view1.data, view2.data):
            if el1[0] != el2[0]:
                raise self.failureException("Mismatched annotation types.")
            if el1[0] in ['vline', 'hline']:
                self.compare_floats(el1[1], el2[1], 'V/H line position')
                self.compare_intervals(el1[2], el2[2])
            elif el1[0] in ['<', '^', '>', 'v']:
                (text1, xy1, points1, arrowstyle1, interval1) = el1[1:]
                (text2, xy2, points2, arrowstyle2, interval2) = el2[1:]

                self.assertEqual(text1, text2, 'Mismatched text in annotation.')
                self.compare_floats(xy1[0], xy2[0], 'Mismatched annotation x position.')
                self.compare_floats(xy1[1], xy2[1], 'Mismatched annotation y position.')
                self.compare_floats(points1, points2,'Mismatched text in annotation.')
                self.assertEqual(arrowstyle1, arrowstyle2, 'Mismatched annotation arrow styles.')
                self.compare_intervals(interval1, interval2)
            else:
                raise NotImplementedError


    #============#
    # DataLayers #
    #============#


    def compare_curve(self, view1, view2, msg):
        self.compare_arrays(view1.data, view2.data, 'Curve data')


    def compare_histogram(self, view1, view2, msg):
        self.compare_arrays(view1.edges, view2.edges, "Histogram edges")
        self.compare_arrays(view1.values, view2.values, "Histogram values")


    def compare_raster(self, view1, view2, msg):
        self.compare_arrays(view1.data, view2.data, 'Raster')


    def compare_heatmap(self, view1, view2, msg):
        self.compare_arrays(view1.data, view2.data, 'HeatMap')


    #========#
    # Tables #
    #========#

    def compare_itemtables(self, view1, view2, msg):

        if view1.rows != view2.rows:
            raise self.failureException("ItemTables have different numbers of rows.")

        if view1.cols != view2.cols:
            raise self.failureException("ItemTables have different numbers of columns.")

        if [d.name for d in view1.dimensions] != [d.name for d in view2.dimensions]:
            raise self.failureException("ItemTables have different Dimensions.")


    def compare_tables(self, view1, view2, msg):

        if view1.rows != view2.rows:
            raise self.failureException("Tables have different numbers of rows.")

        if view1.cols != view2.cols:
            raise self.failureException("Tables have different numbers of columns.")

        self.compare_maps(view1, view2, msg)


    #=============#
    # SheetLayers #
    #=============#

    def compare_matrix(self, view1, view2, msg):
        self.compare_arrays(view1.data, view2.data, 'Matrix')
        self.bounds_check(view1,view2)


    def compare_contours(self, view1, view2, msg):
        if len(view1) != len(view2):
            raise self.failureException("Contours do not have a matching number of contours.")

        for c1, c2 in zip(view1.data, view2.data):
            self.compare_arrays(c1, c2, 'Contour data')


    def compare_points(self, view1, view2, msg):
        if len(view1) != len(view2):
            raise self.failureException("Points objects have different numbers of points.")

        self.compare_arrays(view1.data, view2.data, 'Points data')


    def compare_vectorfield(self, view1, view2, msg):
        if len(view1) != len(view2):
            raise self.failureException("VectorField objects have different numbers of vectors.")

        self.compare_arrays(view1.data, view2.data, 'VectorField data')

    #========#
    # Pandas #
    #========#

    def compare_dframe(self, view1, view2, msg):
        from pandas.util.testing import assert_frame_equal
        try:
            assert_frame_equal(view1.data, view2.data)
        except AssertionError as e:
            raise self.failureException(msg+': '+str(e))

    #=========#
    # Seaborn #
    #=========#

    def compare_distribution(self, view1, view2, msg):
        self.compare_arrays(view1.data, view2.data, 'Distribution data')

    def compare_timeseries(self, view1, view2, msg):
        self.compare_arrays(view1.data, view2.data, 'TimeSeries data')

    def compare_bivariate(self, view1, view2, msg):
        self.compare_arrays(view1.data, view2.data, 'Bivariate data')

    def compare_regression(self, view1, view2, msg):
        self.compare_arrays(view1.data, view2.data, 'Regression data')

    #=======#
    # Grids #
    #=======#

    def _compare_grids(self, view1, view2, name):

        if len(view1.keys()) != len(view2.keys()):
            raise self.failureException("%ss have different numbers of items." % name)

        if set(view1.keys()) != set(view2.keys()):
            raise self.failureException("%ss have different keys." % name)

        if len(view1) != len(view2):
            raise self.failureException("%ss have different depths." % name)

        for el1, el2 in zip(view1, view2):
            self.assertEqual(el1, el2)


    def compare_grids(self, view1, view2, msg):
        self._compare_grids(view1, view2, 'AxisLayout')

    #=========#
    # Options #
    #=========#

    def compare_settings(self, settings1, settings2, msg):
        self.assertEqual(settings1.kwargs, settings2.kwargs)


    def compare_channelopts(self, opt1, opt2, msg):
        self.assertEqual(opt1.mode, opt2.mode)
        self.assertEqual(opt1.pattern, opt2.pattern)
        self.assertEqual(opt1.patter, opt2.pattern)

    #============#
    # Dimensions #
    #============#

    def compare_dims(self, dim1, dim2, msg):
        if dim1.name != dim2.name:
            raise self.failureException("Dimension names are mismatched.")
        if dim1.cyclic != dim1.cyclic:
            raise self.failureException("Dimension cyclic declarations mismatched.")
        if dim1.range != dim1.range:
            raise self.failureException("Dimension ranges mismatched.")
        if dim1.type != dim1.type:
            raise self.failureException("Dimension type declarations mismatched.")
        if dim1.unit != dim1.unit:
            raise self.failureException("Dimension unit declarations mismatched.")
        if dim1.format_string != dim1.format_string:
            raise self.failureException("Dimension format string declarations mismatched.")


class IPTestCase(ViewTestCase):
    """
    This class extends ViewTestCase to handle IPython specific objects.
    """

    def setUp(self):
        super(IPTestCase, self).setUp()
        try:
            import IPython
            self.ip = IPython.InteractiveShell()
            if self.ip is None:
                raise TypeError()
        except Exception:
                raise SkipTest("IPython could not be started")

        self.addTypeEqualityFunc(HTML, self.skip_comparison)
        self.addTypeEqualityFunc(SVG,  self.skip_comparison)

    def skip_comparison(self, obj1, obj2, msg):
        pass

    def get_object(self, name):
        obj = self.ip._object_find(name).obj
        if obj is None:
            raise self.failureException("Could not find object %s" % name)
        return obj


    def cell(self, line):
        "Run an IPython cell"
        self.ip.run_cell(line, silent=True)

    def cell_magic(self, *args, **kwargs):
        "Run an IPython cell magic"
        self.ip.run_cell_magic(*args, **kwargs)


    def line_magic(self, *args, **kwargs):
        "Run an IPython line magic"
        self.ip.run_line_magic(*args, **kwargs)
