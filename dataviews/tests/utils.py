import sys, os
import unittest

from nose.plugins.skip import SkipTest
from numpy.testing import assert_array_almost_equal

cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..', '..'))

from dataviews.views import Layout, GridLayout
from dataviews import DataOverlay,  DataStack,  Annotation, Curve, Histogram
from dataviews import TableStack, Table
from dataviews import SheetOverlay, SheetStack, SheetView, Points, Contours
from dataviews import CoordinateGrid, DataGrid


from dataviews.options import StyleOpts, PlotOpts, ChannelOpts

from IPython.display import HTML, SVG

class ViewTestCase(unittest.TestCase):
    """
    The class implements comparisons between View objects for the
    purposes of testing. The most important attribute that needs to be
    compared is the data attribute as this contains the raw data held
    by the View object.
    """
    def __init__(self, *args, **kwargs):
        super(ViewTestCase, self).__init__(*args, **kwargs)
        # General view classes
        self.addTypeEqualityFunc(GridLayout,   self.compare_gridlayout)
        self.addTypeEqualityFunc(Layout,       self.compare_layouts)
        self.addTypeEqualityFunc(Annotation,   self.compare_annotations)
        # DataLayers
        self.addTypeEqualityFunc(DataOverlay,  self.compare_dataoverlays)
        self.addTypeEqualityFunc(DataStack,    self.compare_datastack)
        self.addTypeEqualityFunc(Curve,        self.compare_curve)
        self.addTypeEqualityFunc(Histogram,    self.compare_histogram)
        # Tables
        self.addTypeEqualityFunc(TableStack,   self.compare_tablestack)
        self.addTypeEqualityFunc(Table,        self.compare_tables)
        # SheetLayers
        self.addTypeEqualityFunc(SheetOverlay, self.compare_sheetoverlays)
        self.addTypeEqualityFunc(SheetStack,   self.compare_sheetstack)
        self.addTypeEqualityFunc(SheetView,    self.compare_sheetviews)
        self.addTypeEqualityFunc(Contours,     self.compare_contours)
        self.addTypeEqualityFunc(Points,       self.compare_points)
        # CoordinateGrid and DataGrid
        self.addTypeEqualityFunc(CoordinateGrid, self.compare_coordgrids)
        self.addTypeEqualityFunc(DataGrid,       self.compare_datagrids)
        # Option objects
        self.addTypeEqualityFunc(StyleOpts, self.compare_opts)
        self.addTypeEqualityFunc(PlotOpts, self.compare_opts)
        self.addTypeEqualityFunc(ChannelOpts, self.compare_channelopts)


    def compare_opts(self, opt1, opt2, msg):
        self.assertEqual(opt1.items, opt2.items)


    def compare_channelopts(self, opt1, opt2, msg):
        self.assertEqual(opt1.mode, opt2.mode)
        self.assertEqual(opt1.pattern, opt2.pattern)
        self.assertEqual(opt1.patter, opt2.pattern)



    #================#
    # Helper methods #
    #================#

    def compare_arrays(self, arr1, arr2, name):
        try:
            assert_array_almost_equal(arr1, arr2)
        except AssertionError as e:
            raise self.failureException(name+': '+str(e))


    def bounds_check(self, view1, view2):
        if view1.bounds.lbrt() != view2.bounds.lbrt():
            raise self.failureException("BoundingBoxes are mismatched.")


    def compare_stack(self, view1, view2, msg):

        if view1.ndims != view2.ndims:
            raise self.failureException("Stacks have different numbers of dimensions.")

        if view1.dimension_labels != view2.dimension_labels:
            raise self.failureException("Stacks have different dimension labels.")

        if len(view1.keys()) != len(view2.keys()):
            raise self.failureException("Stacks have different numbers of keys.")

        if set(view1.keys()) != set(view2.keys()):
            raise self.failureException("Stacks have different sets of keys.")

        for el1, el2 in zip(view1, view2):
            self.assertEqual(el1,el2)

    #=================#
    # Generic classes #
    #=================#

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

    def compare_annotations(self, view1, view2, msg):
        if set(view1.data) != set(view2.data):
            raise self.failureException("Annotations contain different sets of annotations.")

    #============#
    # DataLayers #
    #============#

    def compare_datastack(self, view1, view2, msg):
        self.compare_stack(view1, view2, msg)


    def compare_dataoverlays(self, view1, view2, msg):
        if len(view1) != len(view2):
            raise self.failureException("DataOverlays have different lengths.")

        for (layer1, layer2) in zip(view1, view2):
            self.assertEqual(layer1, layer2)


    def compare_curve(self, view1, view2, msg):
        if view1.cyclic_range != view2.cyclic_range:
            raise self.failureException("Curves do not have matching cyclic_range.")
        self.compare_arrays(view1.data, view2.data, 'Curve data')


    def compare_histogram(self, view1, view2, msg):

        if view1.cyclic_range != view2.cyclic_range:
            raise self.failureException("Histograms do not have matching cyclic_range.")

        self.compare_arrays(view1.edges, view2.edges, "Histogram edges")
        self.compare_arrays(view1.values, view2.values, "Histogram values")

    #========#
    # Tables #
    #========#

    def compare_tablestack(self, view1, view2, msg):
        self.compare_stack(view1, view2, msg)


    def compare_tables(self, view1, view2, msg):

        if view1.rows != view2.rows:
            raise self.failureException("Tables have different numbers of rows.")

        if view1.cols != view2.cols:
            raise self.failureException("Tables have different numbers of columns.")

        if view1.heading_map != view2.heading_map:
            raise self.failureException("Tables have different headings.")

        for heading in view1.heading_values():
            self.assertEqual(view1[heading], view2[heading])

    #=============#
    # SheetLayers #
    #=============#

    def compare_sheetstack(self, view1, view2, msg):
        self.bounds_check(view1,view2)
        self.compare_stack(view1, view2, msg)


    def compare_sheetoverlays(self, view1, view2, msg):
        if len(view1) != len(view2):
            raise self.failureException("SheetOverlays have different lengths.")

        self.bounds_check(view1, view2)

        for (layer1, layer2) in zip(view1, view2):
            self.assertEqual(layer1, layer2)


    def compare_sheetviews(self, view1, view2, msg):
        self.compare_arrays(view1.data, view2.data, 'SheetView')
        self.bounds_check(view1,view2)


    def compare_contours(self, view1, view2, msg):
        self.bounds_check(view1, view2)

        if len(view1) != len(view2):
            raise self.failureException("Contours do not have a matching number of contours.")

        for c1, c2 in zip(view1.data, view2.data):
            self.compare_arrays(c1, c2, 'Contour data')


    def compare_points(self, view1, view2, msg):
        self.bounds_check(view1, view2)

        if len(view1) != len(view2):
            raise self.failureException("Points objects have different numbers of points.")

        self.compare_arrays(view1.data, view2.data, 'Points data')

    #=======#
    # Grids #
    #=======#

    def compare_grids(self, view1, view2, name ):

        if len(view1.keys()) != len(view2.keys()):
            raise self.failureException("%ss have different numbers of items." % name)

        if set(view1.keys()) != set(view2.keys()):
            raise self.failureException("%ss have different keys." % name)

        if len(view1) != len(view2):
            raise self.failureException("%ss have different depths." % name)

        for el1, el2 in zip(view1, view2):
            self.assertEqual(el1, el2)


    def compare_coordgrids(self, view1, view2, msg):
        self.bounds_check(view1, view2)
        self.compare_grids(view1, view2, 'CoordinateGrid')


    def compare_datagrids(self, view1, view2, msg):
        self.compare_grids(view1, view2, 'DataGrid')


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
