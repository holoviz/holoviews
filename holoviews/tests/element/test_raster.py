"""
Unit tests of Raster elements
"""

import numpy as np
import pytest

from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase


class TestRaster(ComparisonTestCase):

    def setUp(self):
        self.array1 = np.array([(0, 1, 2), (3, 4, 5)])

    def test_raster_init(self):
        Raster(self.array1)

    def test_raster_index(self):
        raster = Raster(self.array1)
        self.assertEqual(raster[0, 1], 3)

    def test_raster_sample(self):
        raster = Raster(self.array1)
        self.assertEqual(raster.sample(y=0),
                         Curve(np.array([(0, 0), (1, 1), (2, 2)]),
                               kdims=['x'], vdims=['z']))

    def test_raster_range_masked(self):
        arr = np.random.rand(10,10)-0.5
        arr = np.ma.masked_where(arr<=0, arr)
        rrange = Raster(arr).range(2)
        self.assertEqual(rrange, (np.min(arr), np.max(arr)))


class TestRGB(ComparisonTestCase):

    def setUp(self):
        self.rgb_array = np.random.randint(0, 255, (3, 3, 4))

    def test_construct_from_array_with_alpha(self):
        rgb = RGB(self.rgb_array)
        self.assertEqual(len(rgb.vdims), 4)

    def test_construct_from_tuple_with_alpha(self):
        rgb = RGB(([0, 1, 2], [0, 1, 2], self.rgb_array))
        self.assertEqual(len(rgb.vdims), 4)

    def test_construct_from_xarray_dataset_with_alpha(self):
        xr = pytest.importorskip('xarray')
        xr_dataset = xr.DataArray(
            data=self.rgb_array,
            coords={"y": [0, 1, 2], "x": [0, 1, 2], "band": list("RGBA")}
        ).to_dataset(dim="band")
        rgb = RGB(xr_dataset)
        assert str(rgb.alpha_dimension) in xr_dataset.data_vars
        assert len(rgb.vdims) == 4

    def test_construct_from_dict_with_alpha(self):
        rgb = RGB({'x': [1, 2, 3], 'y': [1, 2, 3], ('R', 'G', 'B', 'A'): self.rgb_array})
        self.assertEqual(len(rgb.vdims), 4)

    def test_not_using_class_variables_vdims(self):
        init_vdims = RGB(self.rgb_array).vdims
        cls_vdims = RGB.vdims
        for i, c in zip(init_vdims, cls_vdims, strict=None):
            assert i is not c
            assert i == c

    def test_nodata(self):
        N = 2
        rgb_d = np.linspace(0, 1, N * N * 3).reshape(N, N, 3)
        rgb = RGB(rgb_d)
        assert sum(np.isnan(rgb["R"])) == 0
        assert sum(np.isnan(rgb["G"])) == 0
        assert sum(np.isnan(rgb["B"])) == 0

        rgb_n = rgb.redim.nodata(R=0)
        assert sum(np.isnan(rgb_n["R"])) == 1
        assert sum(np.isnan(rgb_n["G"])) == 0
        assert sum(np.isnan(rgb_n["B"])) == 0

class TestHSV(ComparisonTestCase):

    def setUp(self):
        self.hsv_array = np.random.randint(0, 255, (3, 3, 4))

    def test_not_using_class_variables_vdims(self):
            init_vdims = HSV(self.hsv_array).vdims
            cls_vdims = HSV.vdims
            for i, c in zip(init_vdims, cls_vdims, strict=None):
                assert i is not c
                assert i == c

class TestQuadMesh(ComparisonTestCase):

    def setUp(self):
        self.array1 = np.array([(0, 1, 2), (3, 4, 5)])

    def test_cast_image_to_quadmesh(self):
        img = Image(self.array1, kdims=['a', 'b'], vdims=['c'], group='A', label='B')
        qmesh = QuadMesh(img)
        self.assertEqual(qmesh.dimension_values(0, False), np.array([-0.333333, 0., 0.333333]))
        self.assertEqual(qmesh.dimension_values(1, False), np.array([-0.25, 0.25]))
        self.assertEqual(qmesh.dimension_values(2, flat=False), self.array1[::-1])
        self.assertEqual(qmesh.kdims, img.kdims)
        self.assertEqual(qmesh.vdims, img.vdims)
        self.assertEqual(qmesh.group, img.group)
        self.assertEqual(qmesh.label, img.label)

    def test_quadmesh_to_trimesh(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
        trimesh = qmesh.trimesh()
        simplices = np.array([[0, 1, 3, 0],
                              [1, 2, 4, 2],
                              [3, 4, 6, 1],
                              [4, 5, 7, 3],
                              [4, 3, 1, 0],
                              [5, 4, 2, 2],
                              [7, 6, 4, 1],
                              [8, 7, 5, 3]])
        vertices = np.array([(-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5),
                             (0.5, -0.5), (0.5, 0.5), (0.5, 1.5),
                             (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)])
        self.assertEqual(trimesh.array(), simplices)
        self.assertEqual(trimesh.nodes.array([0, 1]), vertices)
