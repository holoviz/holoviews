from holoviews import HoloMap, DynamicMap, Curve
from holoviews.core.traversal import unique_dimkeys
from holoviews.element.comparison import ComparisonTestCase


class TestUniqueDimKeys(ComparisonTestCase):

    def test_unique_keys_complete_overlap(self):
        hmap1 = HoloMap({i: Curve(range(10)) for i in range(5)})
        hmap2 = HoloMap({i: Curve(range(10)) for i in range(3, 10)})
        dims, keys = unique_dimkeys(hmap1+hmap2)
        self.assertEqual(hmap1.kdims, dims)
        self.assertEqual(keys, [(i,) for i in range(10)])

    def test_unique_keys_partial_overlap(self):
        hmap1 = HoloMap({(i, j): Curve(range(10)) for i in range(5) for j in range(3)}, kdims=['A', 'B'])
        hmap2 = HoloMap({i: Curve(range(10)) for i in range(5)}, kdims=['A'])
        dims, keys = unique_dimkeys(hmap1+hmap2)
        self.assertEqual(hmap1.kdims, dims)
        self.assertEqual(keys, [(i, j) for i in range(5) for j in list(range(3))])

    def test_unique_keys_no_overlap_exception(self):
        hmap1 = HoloMap({i: Curve(range(10)) for i in range(5)}, kdims=['A'])
        hmap2 = HoloMap({i: Curve(range(10)) for i in range(3, 10)})
        exception = ('When combining HoloMaps into a composite plot '
                     'their dimensions must be subsets of each other.')
        with self.assertRaisesRegex(Exception, exception):
            dims, keys = unique_dimkeys(hmap1+hmap2)

    def test_unique_keys_no_overlap_dynamicmap_uninitialized(self):
        dmap1 = DynamicMap(lambda A: Curve(range(10)), kdims=['A'])
        dmap2 = DynamicMap(lambda B: Curve(range(10)), kdims=['B'])
        dims, keys = unique_dimkeys(dmap1+dmap2)
        self.assertEqual(dims, dmap1.kdims+dmap2.kdims)
        self.assertEqual(keys, [])

    def test_unique_keys_no_overlap_dynamicmap_initialized(self):
        dmap1 = DynamicMap(lambda A: Curve(range(10)), kdims=['A'])
        dmap2 = DynamicMap(lambda B: Curve(range(10)), kdims=['B'])
        dmap1[0]
        dmap2[1]
        dims, keys = unique_dimkeys(dmap1+dmap2)
        self.assertEqual(dims, dmap1.kdims+dmap2.kdims)
        self.assertEqual(keys, [(0, 1)])


