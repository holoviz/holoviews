"""Helper classes for comparing the equality of two HoloViews objects.

These classes are designed to integrate with unittest.TestCase (see
the tests directory) while making equality testing easily accessible
to the user.

For instance, to test if two Matrix objects are equal you can use:

Comparison.assertEqual(matrix1, matrix2)

This will raise an AssertionError if the two matrix objects are not
equal, including information regarding what exactly failed to match.

Note that this functionality could not be provided using comparison
methods on all objects as comparison operators only return Booleans and
thus would not supply any information regarding *why* two elements are
considered different.

"""
import contextlib

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from holoviews import element
from holoviews.core import (
    AdjointLayout,
    Dimension,
    Dimensioned,
    DynamicMap,
    Element,
    Empty,
    GridMatrix,
    GridSpace,
    HoloMap,
    Layout,
    NdLayout,
    NdOverlay,
    Overlay,
)
from holoviews.core.options import Cycle, Options
from holoviews.core.util import datetime_types, dt_to_int, dtype_kind, is_float
from holoviews.core.util.dependencies import _is_installed

_equality_type_funcs = {}


class _Comparison:
    """Class used for comparing two HoloViews objects, including complex
    composite objects. Comparisons are available as classmethods, the
    most general being the assertEqual method that is intended to work
    with any input.

    For instance, to test if two Image objects are equal you can use:

    Comparison.assertEqual(matrix1, matrix2)

    """

    almost_equal = False


    @classmethod
    def register(cls):

        # Float comparisons
        _equality_type_funcs[float] =        cls.compare_floats
        _equality_type_funcs[np.float32] =   cls.compare_floats
        _equality_type_funcs[np.float64] =   cls.compare_floats

        # Numpy array comparison
        _equality_type_funcs[np.ndarray]          = cls.compare_arrays
        _equality_type_funcs[np.ma.masked_array]  = cls.compare_arrays

        # Pandas dataframe comparison
        if _is_installed("pandas"):
            import pandas as pd
            _equality_type_funcs[pd.DataFrame] = cls.compare_dataframe

        # Dimension objects
        _equality_type_funcs[Dimension] =    cls.compare_dimensions
        _equality_type_funcs[Dimensioned] =  cls.compare_dimensioned
        _equality_type_funcs[Element]     =  cls.compare_elements

        # Composition (+ and *)
        _equality_type_funcs[Overlay] =     cls.compare_overlays
        _equality_type_funcs[Layout] =      cls.compare_layouttrees
        _equality_type_funcs[Empty] =       cls.compare_empties

        # Annotations
        _equality_type_funcs[element.VLine] =       cls.compare_vline
        _equality_type_funcs[element.HLine] =       cls.compare_hline
        _equality_type_funcs[element.VSpan] =       cls.compare_vspan
        _equality_type_funcs[element.HSpan] =       cls.compare_hspan
        _equality_type_funcs[element.Spline] =      cls.compare_spline
        _equality_type_funcs[element.Arrow] =       cls.compare_arrow
        _equality_type_funcs[element.Text] =        cls.compare_text
        _equality_type_funcs[element.Div] =         cls.compare_div

        # Path comparisons
        _equality_type_funcs[element.Path] =        cls.compare_paths
        _equality_type_funcs[element.Contours] =    cls.compare_contours
        _equality_type_funcs[element.Polygons] =    cls.compare_polygons
        _equality_type_funcs[element.Box] =         cls.compare_box
        _equality_type_funcs[element.Ellipse] =     cls.compare_ellipse
        _equality_type_funcs[element.Bounds] =      cls.compare_bounds

        # Rasters
        _equality_type_funcs[element.Image] =       cls.compare_image
        _equality_type_funcs[element.ImageStack] =  cls.compare_imagestack
        _equality_type_funcs[element.RGB] =         cls.compare_rgb
        _equality_type_funcs[element.HSV] =         cls.compare_hsv
        _equality_type_funcs[element.Raster] =      cls.compare_raster
        _equality_type_funcs[element.QuadMesh] =    cls.compare_quadmesh
        _equality_type_funcs[element.Surface] =     cls.compare_surface
        _equality_type_funcs[element.HeatMap] =     cls.compare_dataset

        # Geometries
        _equality_type_funcs[element.Segments] =    cls.compare_segments
        _equality_type_funcs[element.Rectangles] =       cls.compare_boxes

        # Charts
        _equality_type_funcs[element.Dataset] =      cls.compare_dataset
        _equality_type_funcs[element.Curve] =        cls.compare_curve
        _equality_type_funcs[element.ErrorBars] =    cls.compare_errorbars
        _equality_type_funcs[element.Spread] =       cls.compare_spread
        _equality_type_funcs[element.Area] =         cls.compare_area
        _equality_type_funcs[element.Scatter] =      cls.compare_scatter
        _equality_type_funcs[element.Scatter3D] =    cls.compare_scatter3d
        _equality_type_funcs[element.TriSurface] =   cls.compare_trisurface
        _equality_type_funcs[element.Histogram] =    cls.compare_histogram
        _equality_type_funcs[element.Bars] =         cls.compare_bars
        _equality_type_funcs[element.Spikes] =       cls.compare_spikes
        _equality_type_funcs[element.BoxWhisker] =   cls.compare_boxwhisker
        _equality_type_funcs[element.VectorField] =  cls.compare_vectorfield

        # Graphs
        _equality_type_funcs[element.Graph] =        cls.compare_graph
        _equality_type_funcs[element.Nodes] =        cls.compare_nodes
        _equality_type_funcs[element.EdgePaths] =    cls.compare_edgepaths
        _equality_type_funcs[element.TriMesh] =      cls.compare_trimesh

        # Tables
        _equality_type_funcs[element.ItemTable] =    cls.compare_itemtables
        _equality_type_funcs[element.Table] =        cls.compare_tables
        _equality_type_funcs[element.Points] =       cls.compare_points

        # Statistical
        _equality_type_funcs[element.Bivariate] =    cls.compare_bivariate
        _equality_type_funcs[element.Distribution] = cls.compare_distribution
        _equality_type_funcs[element.HexTiles] =     cls.compare_hextiles

        # NdMappings
        _equality_type_funcs[NdLayout] =      cls.compare_gridlayout
        _equality_type_funcs[AdjointLayout] = cls.compare_adjointlayouts
        _equality_type_funcs[NdOverlay] =     cls.compare_ndoverlays
        _equality_type_funcs[GridSpace] =     cls.compare_grids
        _equality_type_funcs[GridMatrix] =     cls.compare_grids
        _equality_type_funcs[HoloMap] =       cls.compare_holomap
        _equality_type_funcs[DynamicMap] =    cls.compare_dynamicmap

        # Option objects
        _equality_type_funcs[Options] =     cls.compare_options
        _equality_type_funcs[Cycle] =       cls.compare_cycles

    #=====================#
    # Literal comparisons #
    #=====================#

    @classmethod
    def compare_floats(cls, arr1, arr2, msg='Floats'):
        cls.compare_arrays(arr1, arr2, msg)

    @classmethod
    def compare_arrays(cls, arr1, arr2, msg='Arrays'):
        if cls.almost_equal:
            assert_array_almost_equal(arr1, arr2)
        else:
            assert_array_equal(arr1, arr2)

    @classmethod
    def bounds_check(cls, el1, el2, msg=None):
        lbrt1 = el1.bounds.lbrt()
        lbrt2 = el2.bounds.lbrt()
        for v1, v2 in zip(lbrt1, lbrt2, strict=True):
            if isinstance(v1, datetime_types):
                v1 = dt_to_int(v1)
            if isinstance(v2, datetime_types):
                v2 = dt_to_int(v2)
            # cls.assert_array_almost_equal_fn(v1, v2)
            assert v1 == v2


    #=======================================#
    # Dimension and Dimensioned comparisons #
    #=======================================#


    @classmethod
    def compare_dimensions(cls, dim1, dim2, msg=None):

        # 'Weak' equality semantics
        if dim1.name != dim2.name:
            raise cls.failureException(f"Dimension names mismatched: {dim1.name} != {dim2.name}")
        if dim1.label != dim2.label:
            raise cls.failureException(f"Dimension labels mismatched: {dim1.label} != {dim2.label}")

        # 'Deep' equality of dimension metadata (all parameters)
        dim1_params = dim1.param.values()
        dim2_params = dim2.param.values()

        if set(dim1_params.keys()) != set(dim2_params.keys()):
            raise cls.failureException(f"Dimension parameter sets mismatched: {set(dim1_params.keys())} != {set(dim2_params.keys())}")

        for k in dim1_params.keys():
            if (dim1.param.objects('existing')[k].__class__.__name__ == 'Callable'
                and dim2.param.objects('existing')[k].__class__.__name__ == 'Callable'):
                continue
            try:  # This is needed as two lists are not compared by contents using ==
                cls.assertEqual(dim1_params[k], dim2_params[k], msg=None)
            except AssertionError as e:
                msg = f'Dimension parameter {k!r} mismatched: '
                raise cls.failureException(f"{msg}{e!s}") from e

    @classmethod
    def compare_labelled_data(cls, obj1, obj2, msg=None):
        cls.assertEqual(obj1.group, obj2.group, "Group labels mismatched.")
        cls.assertEqual(obj1.label, obj2.label, "Labels mismatched.")

    @classmethod
    def compare_dimension_lists(cls, dlist1, dlist2, msg='Dimension lists'):
        if len(dlist1) != len(dlist2):
            raise cls.failureException(f'{msg} mismatched')
        for d1, d2 in zip(dlist1, dlist2, strict=None):
            cls.assertEqual(d1, d2)

    @classmethod
    def compare_dimensioned(cls, obj1, obj2, msg=None):
        cls.compare_labelled_data(obj1, obj2)
        cls.compare_dimension_lists(obj1.vdims, obj2.vdims,
                                    'Value dimension list')
        cls.compare_dimension_lists(obj1.kdims, obj2.kdims,
                                    'Key dimension list')

    @classmethod
    def compare_elements(cls, obj1, obj2, msg=None):
        cls.compare_labelled_data(obj1, obj2)
        cls.assertEqual(obj1.data, obj2.data)


    #===============================#
    # Compositional trees (+ and *) #
    #===============================#

    @classmethod
    def compare_trees(cls, el1, el2, msg='Trees'):
        assert el1.keys() == el2.keys()
        for element1, element2 in zip(el1.values(),  el2.values(), strict=None):
            cls.assertEqual(element1, element2)

    @classmethod
    def compare_layouttrees(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        cls.compare_trees(el1, el2, msg='Layouts')

    @classmethod
    def compare_empties(cls, el1, el2, msg=None):
        assert all(isinstance(el, Empty) for el in [el1, el2])

    @classmethod
    def compare_overlays(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        cls.compare_trees(el1, el2, msg='Overlays')


    #================================#
    # AttrTree and Map based classes #
    #================================#

    @classmethod
    def compare_ndmappings(cls, el1, el2, msg='NdMappings'):
        cls.compare_dimensioned(el1, el2)
        if len(el1.keys()) != len(el2.keys()):
            raise cls.failureException(f"{msg} have different numbers of keys.")

        if set(el1.keys()) != set(el2.keys()):
            diff1 = [el for el in el1.keys() if el not in el2.keys()]
            diff2 = [el for el in el2.keys() if el not in el1.keys()]
            raise cls.failureException(f"{msg} have different sets of keys. "
                                       + f"In first, not second {diff1}. "
                                       + f"In second, not first: {diff2}.")

        for element1, element2 in zip(el1, el2, strict=None):
            cls.assertEqual(element1, element2)

    @classmethod
    def compare_holomap(cls, el1, el2, msg='HoloMaps'):
        cls.compare_dimensioned(el1, el2)
        cls.compare_ndmappings(el1, el2, msg)


    @classmethod
    def compare_dynamicmap(cls, el1, el2, msg='DynamicMap'):
        cls.compare_dimensioned(el1, el2)
        cls.compare_ndmappings(el1, el2, msg)


    @classmethod
    def compare_gridlayout(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        assert el1.keys() == el2.keys()

        for element1, element2 in zip(el1, el2, strict=None):
            cls.assertEqual(element1,element2)


    @classmethod
    def compare_ndoverlays(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        if len(el1) != len(el2):
            raise cls.failureException("NdOverlays have different lengths.")

        for (layer1, layer2) in zip(el1, el2, strict=None):
            cls.assertEqual(layer1, layer2)

    @classmethod
    def compare_adjointlayouts(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        for element1, element2 in zip(el1, el1, strict=None):
            cls.assertEqual(element1, element2)


    #=============#
    # Annotations #
    #=============#

    @classmethod
    def compare_annotation(cls, el1, el2, msg='Annotation'):
        cls.compare_dimensioned(el1, el2)
        cls.assertEqual(el1.data, el2.data)

    @classmethod
    def compare_hline(cls, el1, el2, msg='HLine'):
        cls.compare_annotation(el1, el2, msg=msg)

    @classmethod
    def compare_vline(cls, el1, el2, msg='VLine'):
        cls.compare_annotation(el1, el2, msg=msg)

    @classmethod
    def compare_vspan(cls, el1, el2, msg='VSpan'):
        cls.compare_annotation(el1, el2, msg=msg)

    @classmethod
    def compare_hspan(cls, el1, el2, msg='HSpan'):
        cls.compare_annotation(el1, el2, msg=msg)

    @classmethod
    def compare_spline(cls, el1, el2, msg='Spline'):
        cls.compare_annotation(el1, el2, msg=msg)

    @classmethod
    def compare_arrow(cls, el1, el2, msg='Arrow'):
        cls.compare_annotation(el1, el2, msg=msg)

    @classmethod
    def compare_text(cls, el1, el2, msg='Text'):
        cls.compare_annotation(el1, el2, msg=msg)

    @classmethod
    def compare_div(cls, el1, el2, msg='Div'):
        cls.compare_annotation(el1, el2, msg=msg)

    #=======#
    # Paths #
    #=======#

    @classmethod
    def compare_paths(cls, el1, el2, msg='Path'):
        cls.compare_dataset(el1, el2, msg)

        paths1 = el1.split()
        paths2 = el2.split()
        if len(paths1) != len(paths2):
            raise cls.failureException(f"{msg} objects do not have a matching number of paths.")
        for p1, p2 in zip(paths1, paths2, strict=None):
            cls.compare_dataset(p1, p2, f'{msg} data')

    @classmethod
    def compare_contours(cls, el1, el2, msg='Contours'):
        cls.compare_paths(el1, el2, msg=msg)

    @classmethod
    def compare_polygons(cls, el1, el2, msg='Polygons'):
        cls.compare_paths(el1, el2, msg=msg)

    @classmethod
    def compare_box(cls, el1, el2, msg='Box'):
        cls.compare_paths(el1, el2, msg=msg)

    @classmethod
    def compare_ellipse(cls, el1, el2, msg='Ellipse'):
        cls.compare_paths(el1, el2, msg=msg)

    @classmethod
    def compare_bounds(cls, el1, el2, msg='Bounds'):
        cls.compare_paths(el1, el2, msg=msg)


    #========#
    # Charts #
    #========#

    @classmethod
    def compare_dataset(cls, el1, el2, msg='Dataset'):
        cls.compare_dimensioned(el1, el2)
        tabular = not (el1.interface.gridded and el2.interface.gridded)
        dimension_data = [(d, el1.dimension_values(d, expanded=tabular),
                           el2.dimension_values(d, expanded=tabular))
                          for d in el1.kdims]
        dimension_data += [(d, el1.dimension_values(d, flat=tabular),
                            el2.dimension_values(d, flat=tabular))
                            for d in el1.vdims]
        if el1.shape[0] != el2.shape[0]:
            raise AssertionError(f"{msg} not of matching length, {el1.shape[0]} vs. {el2.shape[0]}.")
        for dim, d1, d2 in dimension_data:
            with contextlib.suppress(Exception):
                np.testing.assert_equal(np.asarray(d1), np.asarray(d2))
                continue  # if equal, no need to check further

            if d1.dtype != d2.dtype:
                failure_msg = (
                    f"{msg} {dim.pprint_label} columns have different type. "
                    f"First has type {d1}, and second has type {d2}."
                )
                raise cls.failureException(failure_msg)
            if dtype_kind(d1) in 'SUOV':
                if list(d1) != list(d2):
                    failure_msg = f"{msg} along dimension {dim.pprint_label} not equal."
                    raise cls.failureException(failure_msg)
            else:
                cls.compare_arrays(np.asarray(d1), np.asarray(d2), msg)

    @classmethod
    def compare_curve(cls, el1, el2, msg='Curve'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_errorbars(cls, el1, el2, msg='ErrorBars'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_spread(cls, el1, el2, msg='Spread'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_area(cls, el1, el2, msg='Area'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_scatter(cls, el1, el2, msg='Scatter'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_scatter3d(cls, el1, el2, msg='Scatter3D'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_trisurface(cls, el1, el2, msg='TriSurface'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_histogram(cls, el1, el2, msg='Histogram'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_points(cls, el1, el2, msg='Points'):
        cls.compare_dataset(el1, el2, msg)


    @classmethod
    def compare_vectorfield(cls, el1, el2, msg='VectorField'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_bars(cls, el1, el2, msg='Bars'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_spikes(cls, el1, el2, msg='Spikes'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_boxwhisker(cls, el1, el2, msg='BoxWhisker'):
        cls.compare_dataset(el1, el2, msg)


    #============#
    # Geometries #
    #============#

    @classmethod
    def compare_segments(cls, el1, el2, msg='Segments'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_boxes(cls, el1, el2, msg='Rectangles'):
        cls.compare_dataset(el1, el2, msg)

    #=========#
    # Graphs  #
    #=========#

    @classmethod
    def compare_graph(cls, el1, el2, msg='Graph'):
        cls.compare_dataset(el1, el2, msg)
        cls.compare_nodes(el1.nodes, el2.nodes, msg)
        if el1._edgepaths or el2._edgepaths:
            cls.compare_edgepaths(el1.edgepaths, el2.edgepaths, msg)

    @classmethod
    def compare_trimesh(cls, el1, el2, msg='TriMesh'):
        cls.compare_graph(el1, el2, msg)

    @classmethod
    def compare_nodes(cls, el1, el2, msg='Nodes'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_edgepaths(cls, el1, el2, msg='Nodes'):
        cls.compare_paths(el1, el2, msg)


    #=========#
    # Rasters #
    #=========#

    @classmethod
    def compare_raster(cls, el1, el2, msg='Raster'):
        cls.compare_dimensioned(el1, el2)
        cls.compare_arrays(el1.data, el2.data, msg)

    @classmethod
    def compare_quadmesh(cls, el1, el2, msg='QuadMesh'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_heatmap(cls, el1, el2, msg='HeatMap'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_image(cls, el1, el2, msg='Image'):
        cls.bounds_check(el1,el2)
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_imagestack(cls, el1, el2, msg='ImageStack'):
        cls.bounds_check(el1,el2)
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_rgb(cls, el1, el2, msg='RGB'):
        cls.bounds_check(el1,el2)
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_hsv(cls, el1, el2, msg='HSV'):
        cls.bounds_check(el1,el2)
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_surface(cls, el1, el2, msg='Surface'):
        cls.bounds_check(el1,el2)
        cls.compare_dataset(el1, el2, msg)


    #========#
    # Tables #
    #========#

    @classmethod
    def compare_itemtables(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        if el1.rows != el2.rows:
            raise cls.failureException("ItemTables have different numbers of rows.")

        if el1.cols != el2.cols:
            raise cls.failureException("ItemTables have different numbers of columns.")

        if [d.name for d in el1.vdims] != [d.name for d in el2.vdims]:
            raise cls.failureException("ItemTables have different Dimensions.")


    @classmethod
    def compare_tables(cls, el1, el2, msg='Table'):
        cls.compare_dataset(el1, el2, msg)

    #========#
    # Pandas #
    #========#

    @classmethod
    def compare_dataframe(cls, df1, df2, msg='DFrame'):
        from pandas.testing import assert_frame_equal
        assert_frame_equal(df1, df2)

    #============#
    # Statistics #
    #============#

    @classmethod
    def compare_distribution(cls, el1, el2, msg='Distribution'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_bivariate(cls, el1, el2, msg='Bivariate'):
        cls.compare_dataset(el1, el2, msg)

    @classmethod
    def compare_hextiles(cls, el1, el2, msg='HexTiles'):
        cls.compare_dataset(el1, el2, msg)

    #=======#
    # Grids #
    #=======#

    @classmethod
    def _compare_grids(cls, el1, el2, name):

        if len(el1.keys()) != len(el2.keys()):
            raise cls.failureException(f"{name}s have different numbers of items.")

        if set(el1.keys()) != set(el2.keys()):
            raise cls.failureException(f"{name}s have different keys.")

        if len(el1) != len(el2):
            raise cls.failureException(f"{name}s have different depths.")

        for element1, element2 in zip(el1, el2, strict=True):
            cls.assertEqual(element1, element2)

    @classmethod
    def compare_grids(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        cls._compare_grids(el1, el2, 'GridSpace')

    #=========#
    # Options #
    #=========#

    @classmethod
    def compare_options(cls, options1, options2, msg=None):
        cls.assertEqual(options1.kwargs, options2.kwargs)

    @classmethod
    def compare_cycles(cls, cycle1, cycle2, msg=None):
        cls.assertEqual(cycle1.values, cycle2.values)

    @classmethod
    def _simple_equality(cls,first, second, msg=None):
        assert first == second

    @classmethod
    def assertEqual(cls, first, second, msg=None):
        """Classmethod equivalent to unittest.TestCase method

        """
        if not _equality_type_funcs:
            cls.register()

        asserter = None
        if type(first) is type(second) or (is_float(first) and is_float(second)):
            asserter = _equality_type_funcs.get(type(first))

            if asserter is not None:
                if isinstance(asserter, str):
                    asserter = getattr(cls, asserter)

        if asserter is None:
            asserter = cls._simple_equality

        if msg is None:
            asserter(first, second)
        else:
            asserter(first, second, msg=msg)


class _ComparisonAlmost(_Comparison):
    almost_equal = True


def assert_element_equal(element1, element2):
    # Filter non-holoviews elements
    hv_types = (Element, Layout)
    if not isinstance(element1, hv_types):
        raise TypeError(f"First argument is not an allowed type but a {type(element1).__name__!r}.")
    if not isinstance(element2, hv_types):
        raise TypeError(f"Second argument is not an allowed type but a {type(element2).__name__!r}.")

    _Comparison.assertEqual(element1, element2)


def assert_element_almost_equal(element1, element2):
    # Filter non-holoviews elements
    hv_types = (Element, Layout)
    if not isinstance(element1, hv_types):
        raise TypeError(f"First argument is not an allowed type but a {type(element1).__name__!r}.")
    if not isinstance(element2, hv_types):
        raise TypeError(f"Second argument is not an allowed type but a {type(element2).__name__!r}.")

    _ComparisonAlmost.assertEqual(element1, element2)
