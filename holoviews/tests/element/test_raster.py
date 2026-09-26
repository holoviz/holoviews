"""
Unit tests of Raster elements
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import holoviews as hv
from holoviews.testing import assert_data_equal, assert_element_equal

from .._deps import ds, ds_skip, xr, xr_skip


class TestRaster:
    def setup_method(self):
        self.array1 = np.array([(0, 1, 2), (3, 4, 5)])

    def test_raster_init(self):
        hv.Raster(self.array1)

    def test_raster_index(self):
        raster = hv.Raster(self.array1)
        assert raster[0, 1] == 3

    def test_raster_sample(self):
        raster = hv.Raster(self.array1)
        assert_element_equal(
            raster.sample(y=0),
            hv.Curve(np.array([(0, 0), (1, 1), (2, 2)]), kdims=["x"], vdims=["z"]),
        )

    def test_raster_range_masked(self, rng):
        arr = rng.random((10, 10)) - 0.5
        arr = np.ma.masked_where(arr <= 0, arr)
        rrange = hv.Raster(arr).range(2)
        assert rrange == (np.min(arr), np.max(arr))


class TestRGB:
    def setup_method(self):
        self.rgb_array = np.random.default_rng(1).integers(0, 255, (3, 3, 4))

    def test_construct_from_array_with_alpha(self):
        rgb = hv.RGB(self.rgb_array)
        assert len(rgb.vdims) == 4

    def test_construct_from_tuple_with_alpha(self):
        rgb = hv.RGB(([0, 1, 2], [0, 1, 2], self.rgb_array))
        assert len(rgb.vdims) == 4

    @xr_skip
    def test_construct_from_xarray_dataset_with_alpha(self):
        xr_dataset = xr.DataArray(
            data=self.rgb_array, coords={"y": [0, 1, 2], "x": [0, 1, 2], "band": list("RGBA")}
        ).to_dataset(dim="band")
        rgb = hv.RGB(xr_dataset)
        assert str(rgb.alpha_dimension) in xr_dataset.data_vars
        assert len(rgb.vdims) == 4

    def test_construct_from_dict_with_alpha(self):
        rgb = hv.RGB({"x": [1, 2, 3], "y": [1, 2, 3], ("R", "G", "B", "A"): self.rgb_array})
        assert len(rgb.vdims) == 4

    def test_not_using_class_variables_vdims(self):
        init_vdims = hv.RGB(self.rgb_array).vdims
        cls_vdims = hv.RGB.vdims
        assert len(init_vdims) == 4
        assert len(cls_vdims) == 3
        for i, c in zip(init_vdims, cls_vdims, strict=False):
            assert i is not c
            assert i == c

    def test_nodata(self):
        N = 2
        rgb_d = np.linspace(0, 1, N * N * 3).reshape(N, N, 3)
        rgb = hv.RGB(rgb_d)
        assert sum(np.isnan(rgb["R"])) == 0
        assert sum(np.isnan(rgb["G"])) == 0
        assert sum(np.isnan(rgb["B"])) == 0

        rgb_n = rgb.redim.nodata(R=0)
        assert sum(np.isnan(rgb_n["R"])) == 1
        assert sum(np.isnan(rgb_n["G"])) == 0
        assert sum(np.isnan(rgb_n["B"])) == 0


class TestHSV:
    def setup_method(self):
        self.hsv_array = np.random.default_rng(1).integers(0, 255, (3, 3, 4))

    def test_not_using_class_variables_vdims(self):
        init_vdims = hv.HSV(self.hsv_array).vdims
        cls_vdims = hv.HSV.vdims
        assert len(init_vdims) == 4
        assert len(cls_vdims) == 3
        for i, c in zip(init_vdims, cls_vdims, strict=False):
            assert i is not c
            assert i == c


class TestImageStack:
    def setup_method(self):
        self.x = np.arange(3)
        self.y = np.arange(2)
        self.a = np.random.default_rng(1).random((2, 3))
        self.b = np.random.default_rng(2).random((2, 3))
        self.c = np.random.default_rng(3).random((2, 3))

    def _constructors(self):
        x, y, a, b, c = self.x, self.y, self.a, self.b, self.c
        return {
            "tuple": hv.ImageStack((x, y, a, b, c), kdims=["x", "y"], vdims=["a", "b", "c"]),
            "list": hv.ImageStack([a, b, c], vdims=["a", "b", "c"]),
            "dict": hv.ImageStack({"x": x, "y": y, "a": a, "b": b, "c": c}),
            "ndarray": hv.ImageStack(np.dstack([a, b, c]), vdims=["a", "b", "c"]),
        }

    @pytest.mark.parametrize("name", ["tuple", "list", "dict", "ndarray"])
    def test_select_vdims_grid_backed(self, name):
        img = self._constructors()[name]
        sel = img.select(vdims=["a", "c"])
        assert [vd.name for vd in sel.vdims] == ["a", "c"]
        assert_data_equal(
            sel.dimension_values("a", flat=False), img.dimension_values("a", flat=False)
        )
        assert_data_equal(
            sel.dimension_values("c", flat=False), img.dimension_values("c", flat=False)
        )

    @pytest.mark.parametrize("name", ["tuple", "list", "dict", "ndarray"])
    def test_select_single_vdim_grid_backed(self, name):
        img = self._constructors()[name]
        sel = img.select(vdims="b")
        assert [vd.name for vd in sel.vdims] == ["b"]

    @xr_skip
    def test_select_vdims_xarray_dataset(self):
        ds_ = xr.Dataset(
            {"a": (("y", "x"), self.a), "b": (("y", "x"), self.b), "c": (("y", "x"), self.c)},
            coords={"x": self.x, "y": self.y},
        )
        img = hv.ImageStack(ds_, kdims=["x", "y"])
        sel = img.select(vdims=["a", "c"])
        assert [vd.name for vd in sel.vdims] == ["a", "c"]

    @xr_skip
    def test_select_vdims_xarray_dataarray_packed(self):
        ds_ = xr.Dataset(
            {"a": (("y", "x"), self.a), "b": (("y", "x"), self.b), "c": (("y", "x"), self.c)},
            coords={"x": self.x, "y": self.y},
        ).to_array("level")
        img = hv.ImageStack(ds_, vdims=["level"])
        sel = img.select(vdims=["a", "c"])
        assert [vd.name for vd in sel.vdims] == ["a", "c"]

    def test_select_vdims_missing_raises(self):
        img = self._constructors()["tuple"]
        with pytest.raises(KeyError):
            img.select(vdims=["nope"])

    def test_select_kdim_and_vdim_combined(self):
        img = self._constructors()["tuple"]
        sel = img.select(x=(0, 1), vdims=["a", "b"])
        assert [vd.name for vd in sel.vdims] == ["a", "b"]

    def test_select_without_vdims_unaffected(self):
        img = self._constructors()["tuple"]
        sel = img.select(x=(0, 1))
        assert [vd.name for vd in sel.vdims] == ["a", "b", "c"]

    @ds_skip
    @xr_skip
    def test_select_vdims_rasterized_categorical(self):
        # Regression test: rasterize with a categorical aggregator produces
        # an xarray-backed ImageStack whose select() used to raise
        # TypeError("Constant parameter 'vdims' cannot be modified")
        from holoviews.operation.datashader import rasterize

        df = pd.DataFrame(
            {
                "x": np.random.default_rng(4).random(100),
                "y": np.random.default_rng(5).random(100),
                "cat": np.random.default_rng(6).choice(list("abc"), 100),
            }
        )
        points = hv.Points(df, kdims=["x", "y"], vdims=["cat"])
        agg = rasterize(points, aggregator=ds.count_cat("cat"), dynamic=False)
        sel = agg.select(vdims=["a", "b"])
        assert [vd.name for vd in sel.vdims] == ["a", "b"]


class TestQuadMesh:
    def setup_method(self):
        self.array1 = np.array([(0, 1, 2), (3, 4, 5)])

    def test_cast_image_to_quadmesh(self):
        img = hv.Image(self.array1, kdims=["a", "b"], vdims=["c"], group="A", label="B")
        qmesh = hv.QuadMesh(img)
        assert_data_equal(qmesh.dimension_values(0, False), np.array([-0.333333, 0.0, 0.333333]))
        assert_data_equal(qmesh.dimension_values(1, False), np.array([-0.25, 0.25]))
        assert_data_equal(qmesh.dimension_values(2, flat=False), self.array1[::-1])
        assert qmesh.kdims == img.kdims
        assert qmesh.vdims == img.vdims
        assert qmesh.group == img.group
        assert qmesh.label == img.label

    def test_quadmesh_to_trimesh(self):
        qmesh = hv.QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
        trimesh = qmesh.trimesh()
        simplices = np.array(
            [
                [0, 1, 3, 0],
                [1, 2, 4, 2],
                [3, 4, 6, 1],
                [4, 5, 7, 3],
                [4, 3, 1, 0],
                [5, 4, 2, 2],
                [7, 6, 4, 1],
                [8, 7, 5, 3],
            ]
        )
        vertices = np.array(
            [
                (-0.5, -0.5),
                (-0.5, 0.5),
                (-0.5, 1.5),
                (0.5, -0.5),
                (0.5, 0.5),
                (0.5, 1.5),
                (1.5, -0.5),
                (1.5, 0.5),
                (1.5, 1.5),
            ]
        )
        assert_data_equal(trimesh.array(), simplices)
        assert_data_equal(trimesh.nodes.array([0, 1]), vertices)
