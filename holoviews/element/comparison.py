"""
Helper classes for comparing the equality of two holoview objects.

These classes are designed to integrate with unittest.TestCase (see
the tests directory) while making equality testing easily accessible
to the user.

For instance, to test if two Matrix objects are equal you can use:

Comparison.assertEqual(matrix1, matrix2)

This will raise an AssertionError if the two matrix objects are not
equal, including information regarding what exactly failed to match.

Note that this functionality could not be provided using comparison
methods on all objects as comparison opertors only return Booleans and
thus would not supply any information regarding *why* two elements are
considered different.
"""

from unittest.util import safe_repr
from numpy.testing import assert_array_almost_equal

from . import *
from ..core import AdjointLayout, Overlay
from ..core.options import Options
from ..interface.pandas import *
from ..interface.seaborn import *


class ComparisonInterface(object):
    """
    This class is designed to allow equality testing to work
    seamlessly with unittest.TestCase as a mix-in by implementing a
    compatible interface (namely the assertEqual method).

    The assertEqual class method is to be overridden by an instance
    method of the same name when used as a mix-in with TestCase. The
    contents of the equality_type_funcs dictionary is suitable for use
    with TestCase.addTypeEqualityFunc.
    """

    equality_type_funcs = {}
    failureException = AssertionError

    @classmethod
    def simple_equality(cls, first, second, msg=None):
        """
        Classmethod equivalent to unittest.TestCase method (longMessage = False.)
        """
        if not first==second:
            standardMsg = '%s != %s' % (safe_repr(first), safe_repr(second))
            raise cls.failureException(msg or standardMsg)


    @classmethod
    def assertEqual(cls, first, second, msg=None):
        """
        Classmethod equivalent to unittest.TestCase method
        """
        asserter = None
        if type(first) is type(second):
            asserter = cls.equality_type_funcs.get(type(first))

            try:              basestring = basestring # Python 2
            except NameError: basestring = str        # Python 3

            if asserter is not None:
                if isinstance(asserter, basestring):
                    asserter = getattr(self, asserter)

        if asserter is None:
            asserter = cls.simple_equality

        if msg is None:
            asserter(first, second)
        else:
            asserter(first, second, msg=msg)


class Comparison(ComparisonInterface):
    """
    Class used for comparing two holoview objects, including complex
    composite objects. Comparisons are available as classmethods, the
    most general being the assertEqual method that is intended to work
    with any input.

    For instance, to test if two Matrix objects are equal you can use:

    Comparison.assertEqual(matrix1, matrix2)
    """

    @classmethod
    def register(cls):

        # Float comparisons
        cls.equality_type_funcs[float] =        cls.compare_floats
        cls.equality_type_funcs[np.float] =     cls.compare_floats
        cls.equality_type_funcs[np.float32] =   cls.compare_floats
        cls.equality_type_funcs[np.float64] =   cls.compare_floats

        # Numpy array comparison
        cls.equality_type_funcs[np.ndarray] =   cls.compare_arrays

        # NdMappings
        cls.equality_type_funcs[NdLayout] =      cls.compare_gridlayout
        cls.equality_type_funcs[AdjointLayout] = cls.compare_layouts
        cls.equality_type_funcs[NdOverlay] =     cls.compare_layers
        cls.equality_type_funcs[Overlay] =       cls.compare_layers
        cls.equality_type_funcs[AxisLayout] =    cls.compare_grids
        cls.equality_type_funcs[HoloMap] =       cls.compare_viewmap

        # Charts
        cls.equality_type_funcs[Curve] =        cls.compare_curve
        cls.equality_type_funcs[Histogram] =    cls.compare_histogram
        cls.equality_type_funcs[Raster] =       cls.compare_raster
        cls.equality_type_funcs[HeatMap] =      cls.compare_heatmap

        # Tables
        cls.equality_type_funcs[ItemTable] =    cls.compare_itemtables
        cls.equality_type_funcs[Table] =        cls.compare_tables

        cls.equality_type_funcs[Contours] =     cls.compare_contours
        cls.equality_type_funcs[Points] =       cls.compare_points
        cls.equality_type_funcs[VectorField] =  cls.compare_vectorfield

        # Rasters
        cls.equality_type_funcs[Matrix] =       cls.compare_matrix


        # Pandas DFrame objects
        cls.equality_type_funcs[DataFrameView] = cls.compare_dframe
        cls.equality_type_funcs[PandasDFrame] =  cls.compare_dframe
        cls.equality_type_funcs[DFrame] =        cls.compare_dframe

        # Seaborn Views
        cls.equality_type_funcs[Bivariate] =    cls.compare_bivariate
        cls.equality_type_funcs[Distribution] = cls.compare_distribution
        cls.equality_type_funcs[Regression] =   cls.compare_regression
        cls.equality_type_funcs[TimeSeries] =   cls.compare_timeseries

        # Option objects
        cls.equality_type_funcs[Options] =     cls.compare_options

        # Dimension objects
        cls.equality_type_funcs[Dimension] =    cls.compare_dims

        return cls.equality_type_funcs


    #================#
    # Helper methods #
    #================#

    @classmethod
    def compare_floats(cls, arr1, arr2, msg='Floats'):
        cls.compare_arrays(arr1, arr2, msg)

    @classmethod
    def compare_arrays(cls, arr1, arr2, name='Arrays'):
        try:
            assert_array_almost_equal(arr1, arr2)
        except AssertionError as e:
            raise cls.failureException(name + str(e)[11:])

    @classmethod
    def bounds_check(cls, view1, view2, msg=None):
        if view1.bounds.lbrt() != view2.bounds.lbrt():
            raise cls.failureException("BoundingBoxes are mismatched.")


    @classmethod
    def compare_maps(cls, view1, view2, msg=None):

        if view1.ndims != view2.ndims:
            raise cls.failureException("Maps have different numbers of dimensions.")

        if [d.name for d in view1.dimensions()] != [d.name for d in view2.dimensions()]:
            raise cls.failureException("Maps have different dimension labels.")

        if len(view1.keys()) != len(view2.keys()):
            raise cls.failureException("Maps have different numbers of keys.")

        if set(view1.keys()) != set(view2.keys()):
            raise cls.failureException("Maps have different sets of keys.")

        for el1, el2 in zip(view1, view2):
            cls.assertEqual(el1,el2)

    #================================#
    # AttrTree and Map based classes #
    #================================#

    @classmethod
    def compare_viewmap(cls, view1, view2, msg=None):
        cls.compare_maps(view1, view2, msg)

    @classmethod
    def compare_gridlayout(cls, view1, view2, msg=None):
        if len(view1) != len(view2):
            raise cls.failureException("GridLayouts have different sizes.")

        if set(view1.keys()) != set(view2.keys()):
            raise cls.failureException("GridLayouts have different keys.")

        for el1, el2 in zip(view1, view2):
            cls.assertEqual(el1,el2)

    @classmethod
    def compare_layouts(cls, view1, view2, msg=None):
        for el1, el2 in zip(view1, view1):
            cls.assertEqual(el1, el2)

    @classmethod
    def compare_layers(cls, view1, view2, msg=None):
        if len(view1) != len(view2):
            raise cls.failureException("Overlays have different lengths.")

        for (layer1, layer2) in zip(view1, view2):
            cls.assertEqual(layer1, layer2)

    #========#
    # Charts #
    #========#

    @classmethod
    def compare_curve(cls, view1, view2, msg=None):
        cls.compare_arrays(view1.data, view2.data, 'Curve data')


    @classmethod
    def compare_histogram(cls, view1, view2, msg=None):
        cls.compare_arrays(view1.edges, view2.edges, "Histogram edges")
        cls.compare_arrays(view1.values, view2.values, "Histogram values")


    @classmethod
    def compare_raster(cls, view1, view2, msg=None):
        cls.compare_arrays(view1.data, view2.data, 'Raster')


    @classmethod
    def compare_heatmap(cls, view1, view2, msg=None):
        cls.compare_arrays(view1.data, view2.data, 'HeatMap')

    @classmethod
    def compare_contours(cls, view1, view2, msg=None):
        if len(view1) != len(view2):
            raise cls.failureException("Contours do not have a matching number of contours.")

        for c1, c2 in zip(view1.data, view2.data):
            cls.compare_arrays(c1, c2, 'Contour data')

    @classmethod
    def compare_points(cls, view1, view2, msg=None):
        if len(view1) != len(view2):
            raise cls.failureException("Points objects have different numbers of points.")

        cls.compare_arrays(view1.data, view2.data, 'Points data')

    @classmethod
    def compare_vectorfield(cls, view1, view2, msg=None):
        if len(view1) != len(view2):
            raise cls.failureException("VectorField objects have different numbers of vectors.")

        cls.compare_arrays(view1.data, view2.data, 'VectorField data')


    #=========#
    # Rasters #
    #=========#

    @classmethod
    def compare_matrix(cls, view1, view2, msg=None):
        cls.compare_arrays(view1.data, view2.data, 'Matrices')
        cls.bounds_check(view1,view2)


    #========#
    # Tables #
    #========#

    @classmethod
    def compare_itemtables(cls, view1, view2, msg=None):

        if view1.rows != view2.rows:
            raise cls.failureException("ItemTables have different numbers of rows.")

        if view1.cols != view2.cols:
            raise cls.failureException("ItemTables have different numbers of columns.")

        if [d.name for d in view1.dimensions()] != [d.name for d in view2.dimensions()]:
            raise cls.failureException("ItemTables have different Dimensions.")


    @classmethod
    def compare_tables(cls, view1, view2, msg=None):

        if view1.rows != view2.rows:
            raise cls.failureException("Tables have different numbers of rows.")

        if view1.cols != view2.cols:
            raise cls.failureException("Tables have different numbers of columns.")

        cls.compare_maps(view1, view2, msg)


    #========#
    # Pandas #
    #========#

    @classmethod
    def compare_dframe(cls, view1, view2, msg=None):
        from pandas.util.testing import assert_frame_equal
        try:
            assert_frame_equal(view1.data, view2.data)
        except AssertionError as e:
            raise cls.failureException(msg+': '+str(e))

    #=========#
    # Seaborn #
    #=========#

    @classmethod
    def compare_distribution(cls, view1, view2, msg=None):
        cls.compare_arrays(view1.data, view2.data, 'Distribution data')

    @classmethod
    def compare_timeseries(cls, view1, view2, msg=None):
        cls.compare_arrays(view1.data, view2.data, 'TimeSeries data')

    @classmethod
    def compare_bivariate(cls, view1, view2, msg=None):
        cls.compare_arrays(view1.data, view2.data, 'Bivariate data')

    @classmethod
    def compare_regression(cls, view1, view2, msg=None):
        cls.compare_arrays(view1.data, view2.data, 'Regression data')

    #=======#
    # Grids #
    #=======#

    @classmethod
    def _compare_grids(cls, view1, view2, name):

        if len(view1.keys()) != len(view2.keys()):
            raise cls.failureException("%ss have different numbers of items." % name)

        if set(view1.keys()) != set(view2.keys()):
            raise cls.failureException("%ss have different keys." % name)

        if len(view1) != len(view2):
            raise cls.failureException("%ss have different depths." % name)

        for el1, el2 in zip(view1, view2):
            cls.assertEqual(el1, el2)

    @classmethod
    def compare_grids(cls, view1, view2, msg=None):
        cls._compare_grids(view1, view2, 'AxisLayout')

    #=========#
    # Options #
    #=========#

    @classmethod
    def compare_options(cls, options1, options2, msg=None):
        cls.assertEqual(options1.kwargs, options2.kwargs)


    @classmethod
    def compare_channelopts(cls, opt1, opt2, msg=None):
        cls.assertEqual(opt1.mode, opt2.mode)
        cls.assertEqual(opt1.pattern, opt2.pattern)
        cls.assertEqual(opt1.patter, opt2.pattern)

    #============#
    # Dimensions #
    #============#

    @classmethod
    def compare_dims(cls, dim1, dim2, msg=None):
        if dim1.name != dim2.name:
            raise cls.failureException("Dimension names are mismatched.")
        if dim1.cyclic != dim1.cyclic:
            raise cls.failureException("Dimension cyclic declarations mismatched.")
        if dim1.range != dim1.range:
            raise cls.failureException("Dimension ranges mismatched.")
        if dim1.type != dim1.type:
            raise cls.failureException("Dimension type declarations mismatched.")
        if dim1.unit != dim1.unit:
            raise cls.failureException("Dimension unit declarations mismatched.")
        if dim1.format_string != dim1.format_string:
            raise cls.failureException("Dimension format string declarations mismatched.")


