"""
Test cases for Dimension and Dimensioned object behaviour.
"""
from holoviews.core import Dimensioned
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
