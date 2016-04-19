from collections import OrderedDict

from holoviews.core import Dimension
from holoviews.core.ndmapping import MultiDimensionalMapping
from holoviews.element.comparison import ComparisonTestCase
from holoviews import HoloMap, Dataset
import numpy as np

class DimensionTest(ComparisonTestCase):

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
        self.assertEqual(dim.pprint_value_string(3.23451), 'Test dimension: 3.2345 Twilight zones')
        self.assertEqual(dim.pprint_value_string(4.23441),  'Test dimension: 4.2344 Twilight zones')


class NdIndexableMappingTest(ComparisonTestCase):

    def setUp(self):
        self.init_items_1D_list = [(1, 'a'), (5, 'b')]
        self.init_item_list = [((1, 2.0), 'a'), ((5, 3.0), 'b')]
        self.init_item_odict = OrderedDict([((1, 2.0), 'a'), ((5, 3.0), 'b')])
        self.dimension_labels = ['intdim', 'floatdim']
        self.dim1 = Dimension('intdim', type=int)
        self.dim2 = Dimension('floatdim', type=float)
        self.time_dimension = Dimension

    def test_idxmapping_init(self):
        MultiDimensionalMapping()

    def test_idxmapping_init_item_odict(self):
        MultiDimensionalMapping(self.init_item_odict, kdims=[self.dim1, self.dim2])

    def test_idxmapping_init_item_list(self):
        MultiDimensionalMapping(self.init_item_list, kdims=[self.dim1, self.dim2])

    def test_idxmapping_init_dimstr(self):
        MultiDimensionalMapping(self.init_item_odict, kdims=self.dimension_labels)

    def test_idxmapping_init_dimensions(self):
        MultiDimensionalMapping(self.init_item_odict, kdims=[self.dim1, self.dim2])

    def test_idxmapping_dimension_labels(self):
        idxmap = MultiDimensionalMapping(self.init_item_odict, kdims=[self.dim1, 'floatdim'])
        self.assertEqual([d.name for d in idxmap.kdims], self.dimension_labels)

    def test_idxmapping_ndims(self):
        dims = [self.dim1, self.dim2, 'strdim']
        idxmap = MultiDimensionalMapping(kdims=dims)
        self.assertEqual(idxmap.ndims, len(dims))

    def test_idxmapping_key_len_check(self):
        try:
            MultiDimensionalMapping(initial_items=self.init_item_odict)
            raise AssertionError('Invalid key length check failed.')
        except KeyError:
            pass

    def test_idxmapping_nested_update(self):
        data1 = [(0, 'a'), (1, 'b')]
        data2 = [(2, 'c'), (3, 'd')]
        data3 = [(2, 'e'), (3, 'f')]

        ndmap1 = MultiDimensionalMapping(data1, kdims=[self.dim1])
        ndmap2 = MultiDimensionalMapping(data2, kdims=[self.dim1])
        ndmap3 = MultiDimensionalMapping(data3, kdims=[self.dim1])

        ndmap_list = [(0.5, ndmap1), (1.5, ndmap2)]
        nested_ndmap = MultiDimensionalMapping(ndmap_list, kdims=[self.dim2])
        nested_ndmap[(0.5,)].update(dict([(0, 'c'), (1, 'd')]))
        self.assertEquals(list(nested_ndmap[0.5].values()), ['c', 'd'])

        nested_ndmap[1.5] = ndmap3
        self.assertEquals(list(nested_ndmap[1.5].values()), ['e', 'f'])

    def test_idxmapping_reindex(self):
        data = [((0, 0.5), 'a'), ((1, 0.5), 'b')]
        ndmap = MultiDimensionalMapping(data, kdims=[self.dim1, self.dim2])

        reduced_dims = ['intdim']
        reduced_ndmap = ndmap.reindex(reduced_dims)

        self.assertEqual([d.name for d in reduced_ndmap.kdims], reduced_dims)

    def test_idxmapping_add_dimension(self):
        ndmap = MultiDimensionalMapping(self.init_items_1D_list, kdims=[self.dim1])
        ndmap2d = ndmap.add_dimension(self.dim2, 0, 0.5)

        self.assertEqual(list(ndmap2d.keys()), [(0.5, 1), (0.5, 5)])
        self.assertEqual(ndmap2d.kdims, [self.dim2, self.dim1])

    def test_idxmapping_apply_key_type(self):
        data = dict([(0.5, 'a'), (1.5, 'b')])
        ndmap = MultiDimensionalMapping(data, kdims=[self.dim1])

        self.assertEqual(list(ndmap.keys()), [0, 1])

    def test_setitem_nested_1(self):
        nested1 = MultiDimensionalMapping([('B', 1)])
        ndmap = MultiDimensionalMapping([('A', nested1)])
        nested2 = MultiDimensionalMapping([('B', 2)])
        ndmap['A'] = nested2
        self.assertEqual(ndmap['A'], nested2)

    def test_setitem_nested_2(self):
        nested1 = MultiDimensionalMapping([('B', 1)])
        ndmap = MultiDimensionalMapping([('A', nested1)])
        nested2 = MultiDimensionalMapping([('C', 2)])
        nested_clone = nested1.clone()
        nested_clone.update(nested2)
        ndmap.update({'A': nested2})
        self.assertEqual(ndmap['A'].data, nested_clone.data)


class HoloMapTest(ComparisonTestCase):

    def setUp(self):
        self.xs = range(11)
        self.y_ints = [i*2 for i in range(11)]
        self.ys = np.linspace(0, 1, 11)
        self.columns = Dataset(np.column_stack([self.xs, self.y_ints]),
                               kdims=['x'], vdims=['y'])


    def test_columns_collapse_heterogeneous(self):
        collapsed = HoloMap({i: Dataset({'x':self.xs, 'y': self.ys * i},
                                        kdims=['x'], vdims=['y'])
                             for i in range(10)}, kdims=['z']).collapse('z', np.mean)
        expected = Dataset({'x':self.xs, 'y': self.ys * 4.5}, kdims=['x'], vdims=['y'])
        self.compare_dataset(collapsed, expected)


    def test_columns_sample_homogeneous(self):
        samples = self.columns.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 10, 20]))
