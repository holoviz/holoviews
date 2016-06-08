"""
Test cases for Dimension and Dimensioned object behaviour.
"""
from holoviews.core import Dimensioned, Dimension
from holoviews.element.comparison import ComparisonTestCase


class DimensionedTest(ComparisonTestCase):

    def test_dimensioned_init(self):
        Dimensioned('An example of arbitrary data')

    def test_dimensioned_constant_label(self):
        label = 'label'
        view = Dimensioned('An example of arbitrary data', label=label)
        self.assertEqual(view.label, label)
        try:
            view.label = 'another label'
            raise AssertionError("Label should be a constant parameter.")
        except TypeError: pass

    def test_dimensionsed_redim_string(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=['Test'])
        self.assertEqual(redimensioned, dimensioned.redim(x='Test'))

    def test_dimensionsed_redim_dimension(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=['Test'])
        self.assertEqual(redimensioned, dimensioned.redim(x=Dimension('Test')))

    def test_dimensionsed_redim_dict(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=['Test'])
        self.assertEqual(redimensioned, dimensioned.redim(x={'name': 'Test'}))

    def test_dimensionsed_redim_dict_range(self):
        redimensioned = Dimensioned('Arbitrary Data', kdims=['x']).redim(x={'range': (0, 10)})
        self.assertEqual(redimensioned.kdims[0].range, (0, 10))
