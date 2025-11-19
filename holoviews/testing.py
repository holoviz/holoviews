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
import narwhals.stable.v2 as nw
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from holoviews import element
from holoviews.core import (
    AdjointLayout,
    BoundingBox,
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
from holoviews.core.util import dtype_kind, is_float
from holoviews.core.util.dependencies import _is_installed

# from holoviews.core.util import datetime_types, dt_to_int, dtype_kind, is_float
# TODO: Remove message + cls.failureException


class _DataComparison:
    """Class used for comparing two HoloViews objects, including complex
    composite objects. Comparisons are available as classmethods, the
    most general being the assertEqual method that is intended to work
    with any input.

    For instance, to test if two Image objects are equal you can use:

    Comparison.assertEqual(matrix1, matrix2)

    """

    # __tracebackhide__ = True

    equality_funcs = {}

    @classmethod
    def register(cls):
        if cls.equality_funcs:
            return cls.equality_funcs
        # Numpy array comparison
        cls.equality_funcs[np.ndarray]          = cls.compare_arrays
        cls.equality_funcs[np.ma.masked_array]  = cls.compare_arrays

        # Narwhals comparisons
        cls.equality_funcs[nw.Series] = cls.compare_narwhals_series
        cls.equality_funcs[nw.DataFrame] = cls.compare_narwhals_dataframe
        cls.equality_funcs[nw.LazyFrame] = cls.compare_narwhals_dataframe

        # Pandas comparison
        if _is_installed("pandas"):
            import pandas as pd
            cls.equality_funcs[pd.Series] = cls.compare_pandas_series
            cls.equality_funcs[pd.DataFrame] = cls.compare_pandas_dataframe

        # if _is_installed("xarray"):

        if _is_installed("dask"):
            import dask.array as da
            cls.equality_funcs[da.Array] = cls.compare_arrays


        return cls.equality_funcs

    @classmethod
    def compare_floats(cls, n1, n2, msg='Floats'):
        assert np.isclose(n1, n2)

    @classmethod
    def compare_arrays(cls, arr1, arr2, msg='Arrays'):
        if dtype_kind(arr1) in "UOM":
            assert_array_equal(arr1, arr2)
        else:
            assert_array_almost_equal(arr1, arr2)

    @classmethod
    def compare_pandas_series(cls, ser1, ser2, msg='Pandas Series'):
        from pandas.testing import assert_series_equal
        assert_series_equal(ser1, ser2)

    @classmethod
    def compare_pandas_dataframe(cls, df1, df2, msg='Pandas DataFrame'):
        from pandas.testing import assert_frame_equal
        assert_frame_equal(df1, df2)

    @classmethod
    def compare_narwhals_series(cls, ser1, ser2, msg='pandas Series'):
        from narwhals.testing import assert_series_equal
        assert_series_equal(ser1, ser2, check_names=False, check_dtypes=False)

    @classmethod
    def compare_narwhals_dataframe(cls, df1, df2, msg='Narwhals DataFrame'):
        from narwhals.testing import assert_series_equal

        assert df1.implementation == df2.implementation

        schema1 = df1.collect_schema()
        schema2 = df2.collect_schema()
        assert schema1 == schema2

        if isinstance(df1, nw.LazyFrame) and isinstance(df2, nw.LazyFrame):
            df1 = df1.collect()
            df2 = df2.collect()

        for col in schema1:
            assert_series_equal(df1[col], df2[col])

    @classmethod
    def _simple_equality(cls, first, second, msg=None):
        # Doing comparison twice for better error output
        check = first == second
        if isinstance(check, bool):
            assert first == second
        else:
            assert (first == second).all()

    @classmethod
    def assert_equal(cls, first, second, msg=None):
        """Classmethod equivalent to unittest.TestCase method

        """
        asserter = None
        if type(first) is type(second):
            asserter = cls.equality_funcs.get(type(first))

            if asserter is not None:
                if isinstance(asserter, str):
                    asserter = getattr(cls, asserter)

        if is_float(first) and is_float(second):
            asserter = cls.compare_floats

        if asserter is None:
            asserter = cls._simple_equality

        asserter(first, second, msg=msg)


class _ElementComparison(_DataComparison):
    """Class used for comparing two HoloViews objects, including complex
    composite objects. Comparisons are available as classmethods, the
    most general being the assertEqual method that is intended to work
    with any input.

    For instance, to test if two Image objects are equal you can use:

    Comparison.assertEqual(matrix1, matrix2)

    """

    equality_funcs = {}

    @classmethod
    def register(cls):
        if cls.equality_funcs:
            return cls.equality_funcs

        super().register()

        # Dimension objects
        cls.equality_funcs[Dimension] =    cls.compare_dimensions
        cls.equality_funcs[Dimensioned] =  cls.compare_dimensioned
        cls.equality_funcs[Element]     =  cls.compare_elements

        # Composition (+ and *)
        cls.equality_funcs[Overlay] =     cls.compare_overlays
        cls.equality_funcs[Layout] =      cls.compare_layouttrees
        cls.equality_funcs[Empty] =       cls.compare_empties

        # Annotations
        cls.equality_funcs[element.VLine] =       cls.compare_vline
        cls.equality_funcs[element.HLine] =       cls.compare_hline
        cls.equality_funcs[element.VSpan] =       cls.compare_vspan
        cls.equality_funcs[element.HSpan] =       cls.compare_hspan
        cls.equality_funcs[element.Spline] =      cls.compare_spline
        cls.equality_funcs[element.Arrow] =       cls.compare_arrow
        cls.equality_funcs[element.Text] =        cls.compare_text
        cls.equality_funcs[element.Div] =         cls.compare_div

        # Path comparisons
        cls.equality_funcs[element.Path] =        cls.compare_paths
        cls.equality_funcs[element.Contours] =    cls.compare_contours
        cls.equality_funcs[element.Polygons] =    cls.compare_polygons
        cls.equality_funcs[element.Box] =         cls.compare_box
        cls.equality_funcs[element.Ellipse] =     cls.compare_ellipse
        cls.equality_funcs[element.Bounds] =      cls.compare_bounds

        # Rasters
        cls.equality_funcs[element.Image] =       cls.compare_image
        cls.equality_funcs[element.ImageStack] =  cls.compare_imagestack
        cls.equality_funcs[element.RGB] =         cls.compare_rgb
        cls.equality_funcs[element.HSV] =         cls.compare_hsv
        cls.equality_funcs[element.Raster] =      cls.compare_raster
        cls.equality_funcs[element.QuadMesh] =    cls.compare_quadmesh
        cls.equality_funcs[element.Surface] =     cls.compare_surface
        cls.equality_funcs[element.HeatMap] =     cls.compare_dataset

        # Geometries
        cls.equality_funcs[element.Segments] =    cls.compare_segments
        cls.equality_funcs[element.Rectangles] =       cls.compare_boxes

        # Charts
        cls.equality_funcs[element.Dataset] =      cls.compare_dataset
        cls.equality_funcs[element.Curve] =        cls.compare_curve
        cls.equality_funcs[element.ErrorBars] =    cls.compare_errorbars
        cls.equality_funcs[element.Spread] =       cls.compare_spread
        cls.equality_funcs[element.Area] =         cls.compare_area
        cls.equality_funcs[element.Scatter] =      cls.compare_scatter
        cls.equality_funcs[element.Scatter3D] =    cls.compare_scatter3d
        cls.equality_funcs[element.TriSurface] =   cls.compare_trisurface
        cls.equality_funcs[element.Histogram] =    cls.compare_histogram
        cls.equality_funcs[element.Bars] =         cls.compare_bars
        cls.equality_funcs[element.Spikes] =       cls.compare_spikes
        cls.equality_funcs[element.BoxWhisker] =   cls.compare_boxwhisker
        cls.equality_funcs[element.VectorField] =  cls.compare_vectorfield

        # Graphs
        cls.equality_funcs[element.Graph] =        cls.compare_graph
        cls.equality_funcs[element.Nodes] =        cls.compare_nodes
        cls.equality_funcs[element.EdgePaths] =    cls.compare_edgepaths
        cls.equality_funcs[element.TriMesh] =      cls.compare_trimesh

        # Tables
        cls.equality_funcs[element.ItemTable] =    cls.compare_itemtables
        cls.equality_funcs[element.Table] =        cls.compare_tables
        cls.equality_funcs[element.Points] =       cls.compare_points

        # Statistical
        cls.equality_funcs[element.Bivariate] =    cls.compare_bivariate
        cls.equality_funcs[element.Distribution] = cls.compare_distribution
        cls.equality_funcs[element.HexTiles] =     cls.compare_hextiles

        # NdMappings
        cls.equality_funcs[NdLayout] =      cls.compare_gridlayout
        cls.equality_funcs[AdjointLayout] = cls.compare_adjointlayouts
        cls.equality_funcs[NdOverlay] =     cls.compare_ndoverlays
        cls.equality_funcs[GridSpace] =     cls.compare_grids
        cls.equality_funcs[GridMatrix] =     cls.compare_grids
        cls.equality_funcs[HoloMap] =       cls.compare_holomap
        cls.equality_funcs[DynamicMap] =    cls.compare_dynamicmap

        # Option objects
        cls.equality_funcs[Options] =     cls.compare_options
        cls.equality_funcs[Cycle] =       cls.compare_cycles
        cls.equality_funcs[BoundingBox] = cls._simple_equality

        return cls.equality_funcs

    #=====================#
    # Literal comparisons #
    #=====================#

    @classmethod
    def bounds_check(cls, el1, el2, msg=None):
        lbrt1 = el1.bounds.lbrt()
        lbrt2 = el2.bounds.lbrt()
        for v1, v2 in zip(lbrt1, lbrt2, strict=True):
            # if isinstance(v1, datetime_types):
            #     v1 = dt_to_int(v1)
            # if isinstance(v2, datetime_types):
            #     v2 = dt_to_int(v2)
            cls.compare_floats(v1, v2)

    #=======================================#
    # Dimension and Dimensioned comparisons #
    #=======================================#

    @classmethod
    def compare_dimensions(cls, dim1, dim2, msg=None):
        # 'Weak' equality semantics
        assert dim1.name == dim2.name
        assert dim1.label == dim2.label

        # 'Deep' equality of dimension metadata (all parameters)
        dim1_params = dim1.param.values()
        dim2_params = dim2.param.values()
        assert dim1_params.keys() == dim2_params.keys()

        for k in dim1_params.keys():
            dim1_callable = dim1.param.objects('existing')[k].__class__.__name__ == 'Callable'
            dim2_callable = dim2.param.objects('existing')[k].__class__.__name__ == 'Callable'
            if (dim1_callable and dim2_callable):
                continue

            # This is needed as two lists are not compared by contents using ==
            cls.assert_equal(dim1_params[k], dim2_params[k])

    @classmethod
    def compare_labelled_data(cls, obj1, obj2, msg=None):
        cls.assert_equal(obj1.group, obj2.group, "Group labels mismatched.")
        cls.assert_equal(obj1.label, obj2.label, "Labels mismatched.")

    @classmethod
    def compare_dimension_lists(cls, dlist1, dlist2, msg='Dimension lists'):
        assert len(dlist1) == len(dlist2)
        for d1, d2 in zip(dlist1, dlist2, strict=None):
            cls.assert_equal(d1, d2)

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
        cls.assert_equal(obj1.data, obj2.data)


    #===============================#
    # Compositional trees (+ and *) #
    #===============================#

    @classmethod
    def compare_trees(cls, el1, el2, msg='Trees'):
        assert el1.keys() == el2.keys()
        for element1, element2 in zip(el1.values(),  el2.values(), strict=None):
            cls.assert_equal(element1, element2)

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
        assert el1.keys() == el2.keys()
        for element1, element2 in zip(el1, el2, strict=None):
            cls.assert_equal(element1, element2)

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
            cls.assert_equal(element1,element2)


    @classmethod
    def compare_ndoverlays(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        assert len(el1) == len(el2)

        for (layer1, layer2) in zip(el1, el2, strict=None):
            cls.assert_equal(layer1, layer2)

    @classmethod
    def compare_adjointlayouts(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        for element1, element2 in zip(el1, el1, strict=None):
            cls.assert_equal(element1, element2)


    #=============#
    # Annotations #
    #=============#

    @classmethod
    def compare_annotation(cls, el1, el2, msg='Annotation'):
        cls.compare_dimensioned(el1, el2)
        cls.assert_equal(el1.data, el2.data)

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
        assert el1.shape[0] == el2.shape[0]
        for _, d1, d2 in dimension_data:
            cls.assert_equal(d1, d2)
            # with contextlib.suppress(Exception):
            #     np.testing.assert_equal(np.asarray(d1), np.asarray(d2))
            #     continue  # if equal, no need to check further

            # if d1.dtype != d2.dtype:
            #     failure_msg = (
            #         f"{msg} {dim.pprint_label} columns have different type. "
            #         f"First has type {d1}, and second has type {d2}."
            #     )
            #     raise cls.failureException(failure_msg)
            # if dtype_kind(d1) in 'SUOV':
            #     if list(d1) != list(d2):
            #         failure_msg = f"{msg} along dimension {dim.pprint_label} not equal."
            #         raise cls.failureException(failure_msg)
            # else:
            #     cls.compare_arrays(np.asarray(d1), np.asarray(d2), msg)

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
        assert el1.rows == el2.rows
        assert el1.cols == el2.cols
        assert [d.name for d in el1.vdims] == [d.name for d in el2.vdims]

    @classmethod
    def compare_tables(cls, el1, el2, msg='Table'):
        cls.compare_dataset(el1, el2, msg)

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
        assert el1.keys() == len(el2.keys())
        assert len(el1) == len(el2)

        for element1, element2 in zip(el1, el2, strict=True):
            cls.assert_equal(element1, element2)

    @classmethod
    def compare_grids(cls, el1, el2, msg=None):
        cls.compare_dimensioned(el1, el2)
        cls._compare_grids(el1, el2, 'GridSpace')

    #=========#
    # Options #
    #=========#

    @classmethod
    def compare_options(cls, options1, options2, msg=None):
        cls.assert_equal(options1.kwargs, options2.kwargs)

    @classmethod
    def compare_cycles(cls, cycle1, cycle2, msg=None):
        cls.assert_equal(cycle1.values, cycle2.values)

def _ptype(obj):
    type_obj = type(obj)
    mod = type_obj.__module__
    name = type_obj.__name__
    if mod == "builtins":
        return str(name)
    return f"{mod}.{name}"


def assert_data_equal(element1, element2):
    types = tuple(_DataComparison.register())
    assert isinstance(element1, types), f"not valid type {_ptype(element1)!r}"
    assert isinstance(element2, types), f"not valid type {_ptype(element2)!r}"

    _DataComparison.assert_equal(element1, element2)

def assert_element_equal(element1, element2):
    types = tuple(_ElementComparison.register())
    assert isinstance(element1, types), f"not valid type {_ptype(element1)!r}"
    assert isinstance(element2, types), f"not valid type {_ptype(element2)!r}"

    types = tuple(_DataComparison.register())
    assert not isinstance(element1, types), "use 'hv.testing.assert_data_equal' instead"
    assert not isinstance(element2, types), "use 'hv.testing.assert_data_equal' instead"

    _ElementComparison.assert_equal(element1, element2)
