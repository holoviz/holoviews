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
        self.init_item_list = [((1, 2.0), 'a'), ((5, 3.0), 'b')]
        self.init_item_odict = OrderedDict([((1, 2.0), 'a'), ((5, 3.0), 'b')])

    def test_idxmapping_init(self):
        NdIndexableMapping()

    def test_idxmapping_init_item_odict(self):
        NdIndexableMapping(self.init_item_odict, dimensions=['intdim', 'floatdim'])

    def test_idxmapping_init_item_list(self):
        NdIndexableMapping(self.init_item_list, dimensions=['intdim', 'floatdim'])

    def test_idxmapping_init_dimstr(self):
        NdIndexableMapping(self.init_item_odict, dimensions=['intdim', 'floatdim'])

    def test_idxmapping_init_meta(self):
        NdIndexableMapping(self.init_item_odict, dimensions=['intdim', 'floatdim'], metadata={'foo':'bar'})

    def test_idxmapping_init_dimensions(self):
        dim1 = Dimension('intdim', type=int)
        dim2 = Dimension('floatdim', type=float)
        NdIndexableMapping(self.init_item_odict, dimensions=[dim1, dim2])

    def test_idxmapping_dimension_labels(self):
        dim1 = Dimension('intdim', type=int)
        idxmap = NdIndexableMapping(self.init_item_odict, dimensions=[dim1, 'floatdim'])
        self.assertEqual(idxmap.dimension_labels, ['intdim', 'floatdim'])



if __name__ == "__main__":
    import nose
    nose.runmodule()
