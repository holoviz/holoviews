from collections.abc import Collection, Mapping

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


class _DataComparison:
    equality_funcs = {}

    @classmethod
    def register(cls):
        if cls.equality_funcs:
            return cls.equality_funcs

        # Numpy array comparison
        cls.equality_funcs[np.ndarray] = cls.compare_arrays
        cls.equality_funcs[np.ma.masked_array] = cls.compare_arrays

        # Narwhals comparisons
        cls.equality_funcs[nw.Series] = cls.compare_narwhals_series
        cls.equality_funcs[nw.DataFrame] = cls.compare_narwhals_dataframe
        cls.equality_funcs[nw.LazyFrame] = cls.compare_narwhals_dataframe

        # Pandas comparison
        if _is_installed("pandas"):
            import pandas as pd

            cls.equality_funcs[pd.Series] = cls.compare_pandas_series
            cls.equality_funcs[pd.DataFrame] = cls.compare_pandas_dataframe

        if _is_installed("dask"):
            import dask.array as da

            cls.equality_funcs[da.Array] = cls.compare_arrays

        return cls.equality_funcs

    @classmethod
    def compare_simple(cls, first, second):
        assert first == second

    @classmethod
    def compare_collections(cls, c1, c2):
        if isinstance(c1, str):
            assert c1 == c2
            return
        assert len(c1) == len(c2)
        for item1, item2 in zip(c1, c2, strict=True):
            cls.assert_equal(item1, item2)

    @classmethod
    def compare_mappings(cls, m1, m2):
        assert m1.keys() == m2.keys()
        for key in m1:
            cls.assert_equal(m1[key], m2[key])

    @classmethod
    def compare_floats(cls, n1, n2):
        is_numpy = hasattr(n1, "dtype") and hasattr(n2, "dtype")
        if is_numpy:
            cls.compare_arrays(n1, n2)
        else:
            assert np.isclose(n1, n2)

    @classmethod
    def compare_arrays(cls, arr1, arr2):
        if dtype_kind(arr1) in "UOM":
            assert_array_equal(arr1, arr2)
        else:
            assert_array_almost_equal(arr1, arr2)

    @classmethod
    def compare_pandas_series(cls, ser1, ser2):
        from pandas.testing import assert_series_equal

        assert_series_equal(ser1, ser2)

    @classmethod
    def compare_pandas_dataframe(cls, df1, df2):
        from pandas.testing import assert_frame_equal

        assert_frame_equal(df1, df2)

    @classmethod
    def compare_narwhals_series(cls, ser1, ser2):
        from narwhals.testing import assert_series_equal

        assert_series_equal(ser1, ser2, check_names=False, check_dtypes=False)

    @classmethod
    def compare_narwhals_dataframe(cls, df1, df2):
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
    def assert_equal(cls, first, second):
        asserter = None
        if type(first) is type(second):
            asserter = cls.equality_funcs.get(type(first))

        if asserter is None:
            if is_float(first) and is_float(second):
                asserter = cls.compare_floats
            elif isinstance(first, Mapping) and isinstance(second, Mapping):
                asserter = cls.compare_mappings
            elif isinstance(first, Collection) and isinstance(second, Collection):
                asserter = cls.compare_collections
            else:
                asserter = cls.compare_simple

        asserter(first, second)


class _ElementComparison(_DataComparison):
    equality_funcs = {}

    @classmethod
    def register(cls):
        if cls.equality_funcs:
            return cls.equality_funcs

        super().register()

        # Dimension objects
        cls.equality_funcs[Dimension] = cls.compare_dimensions
        cls.equality_funcs[Dimensioned] = cls.compare_dimensioned
        cls.equality_funcs[Element] = cls.compare_elements

        # Composition (+ and *)
        cls.equality_funcs[Overlay] = cls.compare_overlays
        cls.equality_funcs[Layout] = cls.compare_layouttrees
        cls.equality_funcs[Empty] = cls.compare_empties

        # Annotations
        cls.equality_funcs[element.VLine] = cls.compare_vline
        cls.equality_funcs[element.HLine] = cls.compare_hline
        cls.equality_funcs[element.VSpan] = cls.compare_vspan
        cls.equality_funcs[element.HSpan] = cls.compare_hspan
        cls.equality_funcs[element.Spline] = cls.compare_spline
        cls.equality_funcs[element.Arrow] = cls.compare_arrow
        cls.equality_funcs[element.Text] = cls.compare_text
        cls.equality_funcs[element.Div] = cls.compare_div

        # Path comparisons
        cls.equality_funcs[element.Path] = cls.compare_paths
        cls.equality_funcs[element.Contours] = cls.compare_contours
        cls.equality_funcs[element.Polygons] = cls.compare_polygons
        cls.equality_funcs[element.Box] = cls.compare_box
        cls.equality_funcs[element.Ellipse] = cls.compare_ellipse
        cls.equality_funcs[element.Bounds] = cls.compare_bounds

        # Rasters
        cls.equality_funcs[element.Image] = cls.compare_image
        cls.equality_funcs[element.ImageStack] = cls.compare_imagestack
        cls.equality_funcs[element.RGB] = cls.compare_rgb
        cls.equality_funcs[element.HSV] = cls.compare_hsv
        cls.equality_funcs[element.Raster] = cls.compare_raster
        cls.equality_funcs[element.QuadMesh] = cls.compare_quadmesh
        cls.equality_funcs[element.Surface] = cls.compare_surface
        cls.equality_funcs[element.HeatMap] = cls.compare_dataset

        # Geometries
        cls.equality_funcs[element.Segments] = cls.compare_segments
        cls.equality_funcs[element.Rectangles] = cls.compare_boxes

        # Charts
        cls.equality_funcs[element.Dataset] = cls.compare_dataset
        cls.equality_funcs[element.Curve] = cls.compare_curve
        cls.equality_funcs[element.ErrorBars] = cls.compare_errorbars
        cls.equality_funcs[element.Spread] = cls.compare_spread
        cls.equality_funcs[element.Area] = cls.compare_area
        cls.equality_funcs[element.Scatter] = cls.compare_scatter
        cls.equality_funcs[element.Scatter3D] = cls.compare_scatter3d
        cls.equality_funcs[element.TriSurface] = cls.compare_trisurface
        cls.equality_funcs[element.Histogram] = cls.compare_histogram
        cls.equality_funcs[element.Bars] = cls.compare_bars
        cls.equality_funcs[element.Spikes] = cls.compare_spikes
        cls.equality_funcs[element.BoxWhisker] = cls.compare_boxwhisker
        cls.equality_funcs[element.VectorField] = cls.compare_vectorfield

        # Graphs
        cls.equality_funcs[element.Graph] = cls.compare_graph
        cls.equality_funcs[element.Nodes] = cls.compare_nodes
        cls.equality_funcs[element.EdgePaths] = cls.compare_edgepaths
        cls.equality_funcs[element.TriMesh] = cls.compare_trimesh

        # Tables
        cls.equality_funcs[element.ItemTable] = cls.compare_itemtables
        cls.equality_funcs[element.Table] = cls.compare_tables
        cls.equality_funcs[element.Points] = cls.compare_points

        # Statistical
        cls.equality_funcs[element.Bivariate] = cls.compare_bivariate
        cls.equality_funcs[element.Distribution] = cls.compare_distribution
        cls.equality_funcs[element.HexTiles] = cls.compare_hextiles

        # NdMappings
        cls.equality_funcs[NdLayout] = cls.compare_gridlayout
        cls.equality_funcs[AdjointLayout] = cls.compare_adjointlayouts
        cls.equality_funcs[NdOverlay] = cls.compare_ndoverlays
        cls.equality_funcs[GridSpace] = cls.compare_grids
        cls.equality_funcs[GridMatrix] = cls.compare_grids
        cls.equality_funcs[HoloMap] = cls.compare_holomap
        cls.equality_funcs[DynamicMap] = cls.compare_dynamicmap

        # Option objects
        cls.equality_funcs[Options] = cls.compare_options
        cls.equality_funcs[Cycle] = cls.compare_cycles
        cls.equality_funcs[BoundingBox] = cls.compare_simple

        return cls.equality_funcs

    # ===================
    # Literal comparisons
    # ===================

    @classmethod
    def bounds_check(cls, el1, el2):
        lbrt1 = el1.bounds.lbrt()
        lbrt2 = el2.bounds.lbrt()
        for v1, v2 in zip(lbrt1, lbrt2, strict=True):
            cls.compare_floats(v1, v2)

    # =====================================
    # Dimension and Dimensioned comparisons
    # =====================================

    @classmethod
    def compare_dimensions(cls, dim1, dim2):
        # 'Weak' equality semantics
        assert dim1.name == dim2.name
        assert dim1.label == dim2.label

        # 'Deep' equality of dimension metadata (all parameters)
        dim1_params = dim1.param.values()
        dim2_params = dim2.param.values()
        assert dim1_params.keys() == dim2_params.keys()

        for k in dim1_params.keys():
            dim1_callable = (
                dim1.param.objects("existing")[k].__class__.__name__ == "Callable"
            )
            dim2_callable = (
                dim2.param.objects("existing")[k].__class__.__name__ == "Callable"
            )
            if dim1_callable and dim2_callable:
                continue

            # This is needed as two lists are not compared by contents using ==
            cls.assert_equal(dim1_params[k], dim2_params[k])

    @classmethod
    def compare_labelled_data(cls, obj1, obj2):
        cls.assert_equal(obj1.group, obj2.group)
        cls.assert_equal(obj1.label, obj2.label)

    @classmethod
    def compare_dimension_lists(cls, dlist1, dlist2):
        assert len(dlist1) == len(dlist2)
        for d1, d2 in zip(dlist1, dlist2, strict=True):
            cls.assert_equal(d1, d2)

    @classmethod
    def compare_dimensioned(cls, obj1, obj2):
        cls.compare_labelled_data(obj1, obj2)
        cls.compare_dimension_lists(obj1.vdims, obj2.vdims)
        cls.compare_dimension_lists(obj1.kdims, obj2.kdims)

    @classmethod
    def compare_elements(cls, obj1, obj2):
        cls.compare_labelled_data(obj1, obj2)
        cls.assert_equal(obj1.data, obj2.data)

    # =============================
    # Compositional trees (+ and *)
    # =============================

    @classmethod
    def compare_trees(cls, el1, el2):
        assert el1.keys() == el2.keys()
        for element1, element2 in zip(el1.values(), el2.values(), strict=True):
            cls.assert_equal(element1, element2)

    @classmethod
    def compare_layouttrees(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        cls.compare_trees(el1, el2)

    @classmethod
    def compare_empties(cls, el1, el2):
        assert all(isinstance(el, Empty) for el in [el1, el2])

    @classmethod
    def compare_overlays(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        cls.compare_trees(el1, el2)

    # ==============================
    # AttrTree and Map based classes
    # ==============================

    @classmethod
    def compare_ndmappings(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        assert el1.keys() == el2.keys()
        for element1, element2 in zip(el1, el2, strict=True):
            cls.assert_equal(element1, element2)

    @classmethod
    def compare_holomap(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        cls.compare_ndmappings(el1, el2)

    @classmethod
    def compare_dynamicmap(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        cls.compare_ndmappings(el1, el2)

    @classmethod
    def compare_gridlayout(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        assert el1.keys() == el2.keys()

        for element1, element2 in zip(el1, el2, strict=True):
            cls.assert_equal(element1, element2)

    @classmethod
    def compare_ndoverlays(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        assert len(el1) == len(el2)

        for layer1, layer2 in zip(el1, el2, strict=True):
            cls.assert_equal(layer1, layer2)

    @classmethod
    def compare_adjointlayouts(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        for element1, element2 in zip(el1, el1, strict=True):
            cls.assert_equal(element1, element2)

    # ===========
    # Annotations
    # ===========

    @classmethod
    def compare_annotation(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        cls.assert_equal(el1.data, el2.data)

    @classmethod
    def compare_hline(cls, el1, el2):
        cls.compare_annotation(el1, el2)

    @classmethod
    def compare_vline(cls, el1, el2):
        cls.compare_annotation(el1, el2)

    @classmethod
    def compare_vspan(cls, el1, el2):
        cls.compare_annotation(el1, el2)

    @classmethod
    def compare_hspan(cls, el1, el2):
        cls.compare_annotation(el1, el2)

    @classmethod
    def compare_spline(cls, el1, el2):
        cls.compare_annotation(el1, el2)

    @classmethod
    def compare_arrow(cls, el1, el2):
        cls.compare_annotation(el1, el2)

    @classmethod
    def compare_text(cls, el1, el2):
        cls.compare_annotation(el1, el2)

    @classmethod
    def compare_div(cls, el1, el2):
        cls.compare_annotation(el1, el2)

    # =====
    # Paths
    # =====

    @classmethod
    def compare_paths(cls, el1, el2):
        cls.compare_dataset(el1, el2)

        paths1 = el1.split()
        paths2 = el2.split()
        assert len(paths1) == len(paths2)
        for p1, p2 in zip(paths1, paths2, strict=True):
            cls.compare_dataset(p1, p2)

    @classmethod
    def compare_contours(cls, el1, el2):
        cls.compare_paths(el1, el2)

    @classmethod
    def compare_polygons(cls, el1, el2):
        cls.compare_paths(el1, el2)

    @classmethod
    def compare_box(cls, el1, el2):
        cls.compare_paths(el1, el2)

    @classmethod
    def compare_ellipse(cls, el1, el2):
        cls.compare_paths(el1, el2)

    @classmethod
    def compare_bounds(cls, el1, el2):
        cls.compare_paths(el1, el2)

    # ======
    # Charts
    # ======

    @classmethod
    def compare_dataset(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        tabular = not (el1.interface.gridded and el2.interface.gridded)
        dimension_data = [
            (
                el1.dimension_values(d, expanded=tabular),
                el2.dimension_values(d, expanded=tabular),
            )
            for d in el1.kdims
        ]
        dimension_data += [
            (
                el1.dimension_values(d, flat=tabular),
                el2.dimension_values(d, flat=tabular),
            )
            for d in el1.vdims
        ]
        assert el1.shape[0] == el2.shape[0]
        for d1, d2 in dimension_data:
            cls.assert_equal(d1, d2)

    @classmethod
    def compare_curve(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_errorbars(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_spread(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_area(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_scatter(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_scatter3d(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_trisurface(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_histogram(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_points(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_vectorfield(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_bars(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_spikes(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_boxwhisker(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    # ==========
    # Geometries
    # ==========

    @classmethod
    def compare_segments(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_boxes(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    # =======
    # Graphs
    # =======

    @classmethod
    def compare_graph(cls, el1, el2):
        cls.compare_dataset(el1, el2)
        cls.compare_nodes(el1.nodes, el2.nodes)
        if el1._edgepaths or el2._edgepaths:
            cls.compare_edgepaths(el1.edgepaths, el2.edgepaths)

    @classmethod
    def compare_trimesh(cls, el1, el2):
        cls.compare_graph(el1, el2)

    @classmethod
    def compare_nodes(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_edgepaths(cls, el1, el2):
        cls.compare_paths(el1, el2)

    # =======
    # Rasters
    # =======

    @classmethod
    def compare_raster(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        cls.compare_arrays(el1.data, el2.data)

    @classmethod
    def compare_quadmesh(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_heatmap(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_image(cls, el1, el2):
        cls.bounds_check(el1, el2)
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_imagestack(cls, el1, el2):
        cls.bounds_check(el1, el2)
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_rgb(cls, el1, el2):
        cls.bounds_check(el1, el2)
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_hsv(cls, el1, el2):
        cls.bounds_check(el1, el2)
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_surface(cls, el1, el2):
        cls.bounds_check(el1, el2)
        cls.compare_dataset(el1, el2)

    # ======
    # Tables
    # ======

    @classmethod
    def compare_itemtables(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        assert el1.rows == el2.rows
        assert el1.cols == el2.cols
        assert [d.name for d in el1.vdims] == [d.name for d in el2.vdims]

    @classmethod
    def compare_tables(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    # ==========
    # Statistics
    # ==========

    @classmethod
    def compare_distribution(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_bivariate(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    @classmethod
    def compare_hextiles(cls, el1, el2):
        cls.compare_dataset(el1, el2)

    # =====
    # Grids
    # =====

    @classmethod
    def _compare_grids(cls, el1, el2):
        assert el1.keys() == el2.keys()
        assert len(el1) == len(el2)
        for element1, element2 in zip(el1, el2, strict=True):
            cls.assert_equal(element1, element2)

    @classmethod
    def compare_grids(cls, el1, el2):
        cls.compare_dimensioned(el1, el2)
        cls._compare_grids(el1, el2)

    # =======
    # Options
    # =======

    @classmethod
    def compare_options(cls, options1, options2):
        cls.assert_equal(options1.kwargs, options2.kwargs)

    @classmethod
    def compare_cycles(cls, cycle1, cycle2):
        cls.assert_equal(cycle1.values, cycle2.values)


def _ptype(obj):
    type_obj = type(obj)
    mod = type_obj.__module__
    name = type_obj.__name__
    if mod == "builtins":
        return str(name)
    return f"{mod}.{name}"


def assert_dict_equal(element1, element2):
    """Asserts that two dictionaries are equal."""
    assert isinstance(element1, Mapping)
    assert isinstance(element2, Mapping)
    _ElementComparison.register()
    _ElementComparison.compare_mappings(element1, element2)


def assert_data_equal(element1, element2):
    """Asserts that two data sources are equal."""
    # For convenience we convert one array and one list into numpy arrays
    if isinstance(element2, np.ndarray) and isinstance(element1, list):
        element1 = np.array(element1)
    if isinstance(element1, np.ndarray) and isinstance(element2, list):
        element2 = np.array(element2)

    types = tuple(_DataComparison.register())
    assert isinstance(element1, types), f"not valid type {_ptype(element1)!r}"
    assert isinstance(element2, types), f"not valid type {_ptype(element2)!r}"

    _DataComparison.assert_equal(element1, element2)


def assert_element_equal(element1, element2):
    """Asserts that two HoloViews Elements are equal."""
    types = tuple(_ElementComparison.register())
    assert isinstance(element1, types), f"not valid type {_ptype(element1)!r}"
    assert isinstance(element2, types), f"not valid type {_ptype(element2)!r}"

    types = tuple(_DataComparison.register())
    assert not isinstance(element1, types), "use 'hv.testing.assert_data_equal' instead"
    assert not isinstance(element2, types), "use 'hv.testing.assert_data_equal' instead"

    _ElementComparison.assert_equal(element1, element2)
