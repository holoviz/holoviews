import unittest

from dataviews.ndmapping import Dimension, NdIndexableMapping
from collections import OrderedDict

class DimensionTest(unittest.TestCase):

    def test_dimension_init(self):
        Dimension('Test dimension')
        Dimension('Test dimension', cyclic=True)
        Dimension('Test dimension', cyclic=True, type=int)
        Dimension('Test dimension', cyclic=True, type=int, unit='Twilight zones')

    def test_dimension_call(self):
        dim1 = Dimension('Test dimension')
        dim2 = dim1(cyclic=True)
        self.assertEqual(dim2.cyclic,True)

        dim3 = dim1('New test dimension', unit='scovilles')
        self.assertEqual(dim3.name, 'New test dimension')
        self.assertEqual(dim3.unit, 'scovilles')

    def test_dimension_pprint(self):
        dim = Dimension('Test dimension', cyclic=True, type=float, unit='Twilight zones')
        self.assertEqual(dim.pprint_value(3.2345), 'Test dimension = 3.23')
        self.assertEqual(dim.pprint_value(4.2345, rounding=3), 'Test dimension = 4.234')


class NdIndexableMappingTest(unittest.TestCase):

    def setUp(self):
        self.init_items_1D_list = [(1, 'a'), (5, 'b')]
        self.init_item_list = [((1, 2.0), 'a'), ((5, 3.0), 'b')]
        self.init_item_odict = OrderedDict([((1, 2.0), 'a'), ((5, 3.0), 'b')])
        self.dimension_labels = ['intdim', 'floatdim']
        self.dim1 = Dimension('intdim', type=int)
        self.dim2 = Dimension('floatdim', type=float)
        self.time_dimension = Dimension

    def test_idxmapping_init(self):
        NdIndexableMapping()

    def test_idxmapping_init_item_odict(self):
        NdIndexableMapping(self.init_item_odict, dimensions=[self.dim1, self.dim2])

    def test_idxmapping_init_item_list(self):
        NdIndexableMapping(self.init_item_list, dimensions=[self.dim1, self.dim2])

    def test_idxmapping_init_dimstr(self):
        NdIndexableMapping(self.init_item_odict, dimensions=self.dimension_labels)

    def test_idxmapping_init_dimensions(self):
        NdIndexableMapping(self.init_item_odict, dimensions=[self.dim1, self.dim2])

    def test_idxmapping_dimension_labels(self):
        idxmap = NdIndexableMapping(self.init_item_odict, dimensions=[self.dim1, 'floatdim'])
        self.assertEqual(idxmap.dimension_labels, self.dimension_labels)

    def test_idxmapping_dim_dict(self):
        idxmap = NdIndexableMapping(dimensions=[self.dim1, self.dim2])
        dim_labels, dim_objs = zip(*idxmap.dim_dict.items())
        self.assertEqual(dim_labels, tuple(self.dimension_labels))
        self.assertEqual(dim_objs, (self.dim1, self.dim2))

    def test_idxmapping_ndims(self):
        dims = [self.dim1, self.dim2, 'strdim']
        idxmap = NdIndexableMapping(dimensions=dims)
        self.assertEqual(idxmap.ndims, len(dims))

    def test_idxmapping_timestamp(self):
        idxmap = NdIndexableMapping(self.init_item_odict,
                                    dimensions=[self.dim1, self.dim2],
                                    timestamp=0.0)
        self.assertEqual(idxmap.metadata.timestamp, 0.0)

    def test_idxmapping_data_type_check(self):
        NdIndexableMapping(self.init_item_odict, data_type=str,
                           dimensions=[self.dim1, self.dim2])
        try:
            NdIndexableMapping(self.init_item_odict, data_type=float,
                               dimensions=[self.dim1, self.dim2])
            raise AssertionError('Data type check failed.')
        except TypeError:
            pass

    def test_idxmapping_key_len_check(self):
        try:
            NdIndexableMapping(initial_items=self.init_item_odict)
            raise AssertionError('Invalid key length check failed.')
        except KeyError:
            pass

    def test_idxmapping_nested_update(self):
        data1 = [(0, 'a'), (1, 'b')]
        data2 = [(2, 'c'), (3, 'd')]
        data3 = [(2, 'e'), (3, 'f')]

        ndmap1 = NdIndexableMapping(data1, dimensions=[self.dim1])
        ndmap2 = NdIndexableMapping(data2, dimensions=[self.dim1])
        ndmap3 = NdIndexableMapping(data3, dimensions=[self.dim1])

        ndmap_list = [(0.5, ndmap1), (1.5, ndmap2)]
        nested_ndmap = NdIndexableMapping(ndmap_list, dimensions=[self.dim2])
        nested_ndmap[(0.5,)].update(dict([(0, 'c'), (1, 'd')]))
        self.assertEquals(nested_ndmap[0.5].values(), ['c', 'd'])

        nested_ndmap[1.5] = ndmap3
        self.assertEquals(nested_ndmap[1.5].values(), ['e', 'f'])

    def test_idxmapping_reindex(self):
        data = [((0, 0.5), 'a'), ((1, 0.5), 'b')]
        ndmap = NdIndexableMapping(data, dimensions=[self.dim1, self.dim2])

        reduced_dims = ['intdim']
        reduced_ndmap = ndmap.reindex(reduced_dims)

        self.assertEqual(reduced_ndmap.dimension_labels, reduced_dims)

    def test_idxmapping_adddimension(self):
        ndmap = NdIndexableMapping(self.init_items_1D_list, dimensions=[self.dim1])
        ndmap2d = ndmap.add_dimension(self.dim2, 0, 0.5)

        self.assertEqual(ndmap2d.keys(), [(0.5, 1), (0.5, 5)])
        self.assertEqual(ndmap2d.dimensions, [self.dim2, self.dim1])

    def test_idxmapping_clone(self):
        ndmap = NdIndexableMapping(self.init_items_1D_list, dimensions=[self.dim1])
        cloned_ndmap = ndmap.clone(data_type=str)

        self.assertEqual(cloned_ndmap.data_type, str)

    def test_idxmapping_apply_key_type(self):
        data = dict([(0.5, 'a'), (1.5, 'b')])
        ndmap = NdIndexableMapping(data, dimensions=[self.dim1])

        self.assertEqual(ndmap.keys(), [0, 1])


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
