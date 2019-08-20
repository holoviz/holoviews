from holoviews.element.comparison import ComparisonTestCase
import pandas as pd
from holoviews import Dataset, Curve, Dimension, Scatter, Histogram


class DatasetPropertyTestCase(ComparisonTestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'a': [1, 1, 3, 3, 2, 2, 0, 0],
            'b': [10, 20, 30, 40, 10, 20, 30, 40],
            'c': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
            'd': [-1, -2, -3, -4, -5, -6, -7, -8]
        })

        self.ds = Dataset(
            self.df,
            kdims=[
                Dimension('a', label="The a Column"),
                Dimension('b', label="The b Column"),
                Dimension('c', label="The c Column"),
                Dimension('d', label="The d Column"),
            ]
        )


class ConstructorTestCase(DatasetPropertyTestCase):
    def test_constructors(self):
        expected = Dataset(self.df)
        self.assertIs(expected, expected.dataset)

        element = Curve(self.df)
        self.assertEqual(element.dataset, expected)
