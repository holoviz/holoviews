from holoviews.element.comparison import ComparisonTestCase
import pandas as pd
import numpy as np
from holoviews import Dataset, Curve, Dimension, Scatter, Distribution
import dask.dataframe as dd
from holoviews.operation import histogram
from holoviews import dim

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
    def test_constructors_dataset(self):
        expected = Dataset(self.df)
        self.assertIs(expected, expected.dataset)

    def test_constructor_curve(self):
        element = Curve(self.df)
        expected = Dataset(self.df)
        self.assertEqual(element.dataset, expected)


class ToTestCase(DatasetPropertyTestCase):
    def test_to_element(self):
        curve = self.ds.to(Curve, 'a', 'b', groupby=[])
        self.assertEqual(curve.dataset, self.ds)

        scatter = curve.to(Scatter)
        self.assertEqual(scatter.dataset, self.ds)

    def test_to_holomap(self):
        curve_hmap = self.ds.to(Curve, 'a', 'b', groupby=['c'])

        # Check HoloMap element datasets
        for v in self.df.c.drop_duplicates():
            curve = curve_hmap.data[(v,)]
            self.assertEqual(
                curve.dataset, self.ds.select(c=v)
            )

    def test_to_holomap_dask(self):
        ddf = dd.from_pandas(self.df, npartitions=2)
        dds = Dataset(
            ddf,
            kdims=[
                Dimension('a', label="The a Column"),
                Dimension('b', label="The b Column"),
                Dimension('c', label="The c Column"),
                Dimension('d', label="The d Column"),
            ]
        )

        curve_hmap = dds.to(Curve, 'a', 'b', groupby=['c'])

        # Check HoloMap element datasets
        for v in self.df.c.drop_duplicates():
            curve = curve_hmap.data[(v,)]
            self.assertEqual(
                curve.dataset, self.ds.select(c=v)
            )


class CloneTestCase(DatasetPropertyTestCase):
    def test_clone(self):
        # Dataset
        self.assertEqual(self.ds.clone().dataset, self.ds)

        # Curve
        self.assertEqual(
            self.ds.to.curve('a', 'b', groupby=[]).clone().dataset,
            self.ds
        )


class ReindexTestCase(DatasetPropertyTestCase):
    def test_reindex_dataset(self):
        ds_ab = self.ds.reindex(kdims=['a'], vdims=['b'])
        self.assertEqual(ds_ab.dataset, self.ds)

    def test_double_reindex_dataset(self):
        ds_abc = self.ds.reindex(kdims=['a'], vdims=['b', 'c'])
        ds_ab = ds_abc.reindex(kdims=['a'], vdims=['b'])
        self.assertEqual(ds_ab.dataset, self.ds)

    def test_reindex_curve(self):
        curve_ab = self.ds.to(Curve, 'a', 'b', groupby=[])
        curve_ba = curve_ab.reindex(kdims='b', vdims='a')
        self.assertEqual(curve_ab.dataset, self.ds)
        self.assertEqual(curve_ba.dataset, self.ds)

    def test_double_reindex_curve(self):
        curve_abc = self.ds.to(Curve, 'a', ['b', 'c'], groupby=[])
        curve_ab = curve_abc.reindex(kdims='a', vdims='b')
        curve_ba = curve_ab.reindex(kdims='b', vdims='a')
        self.assertEqual(curve_ab.dataset, self.ds)
        self.assertEqual(curve_ba.dataset, self.ds)


class IlocTestCase(DatasetPropertyTestCase):
    def test_iloc_dataset(self):
        expected = self.ds.iloc[[0, 2]]

        # Dataset
        self.assertEqual(
            self.ds.clone().iloc[[0, 2]].dataset,
            expected
        )

    def test_iloc_curve(self):
        expected = self.ds.iloc[[0, 2]]

        # Curve
        curve = self.ds.to.curve('a', 'b', groupby=[])
        self.assertEqual(
            curve.iloc[[0, 2]].dataset,
            expected
        )


class SelectTestCase(DatasetPropertyTestCase):
    def test_select_dataset(self):
        # Dataset
        self.assertEqual(
            self.ds.select(b=10).dataset,
            self.ds.select(b=10)
        )

    def test_select_curve(self):
        # Curve
        self.assertEqual(
            self.ds.to.curve('a', 'b', groupby=[]).select(b=10).dataset,
            self.ds.select(b=10)
        )

    def test_select_curve_all_dimensions(self):
        curve1 = self.ds.to.curve('a', 'b', groupby=[])

        # Check curve1 dataset property
        self.assertEqual(curve1.dataset, self.ds)

        # Down select curve 1 on b, which is a value dimension, and c,
        # which is a dimension in the original dataset, but not a kdim or vdim
        curve2 = curve1.select(b=10, c='A')

        # This selection should be equivalent to down selecting the dataset
        # before creating the curve
        self.assertEqual(
            curve2,
            self.ds.select(b=10, c='A').to.curve('a', 'b', groupby=[])
        )

        # Check that we get the same result when using a dim expression
        curve3 = curve1.select((dim('b') == 10) & (dim('c') == 'A'))
        self.assertEqual(curve3, curve2)

class HistogramTestCase(DatasetPropertyTestCase):

    def setUp(self):
        super(HistogramTestCase, self).setUp()
        self.hist = self.ds.hist('a', adjoin=False, normed=False)

    def test_construction(self):
        self.assertEqual(self.hist.dataset, self.ds)

    def test_clone(self):
        self.assertEqual(self.hist.clone().dataset, self.ds)

    def test_select_single(self):
        sub_hist = self.hist.select(a=(1, None))
        self.assertEqual(sub_hist.dataset, self.ds.select(a=(1, None)))

    def test_select_multi(self):
        # Add second selection on b. b is a dimension in hist.dataset but
        # not in hist.  Make sure that we apply the selection on both
        # properties.
        sub_hist = self.hist.select(a=(1, None), b=100)

        self.assertEqual(
            sub_hist.dataset,
            self.ds.select(a=(1, None), b=100)
        )

    def test_hist_to_curve(self):
        # No exception thrown
        self.hist.to.curve()

    def test_hist_selection_all_dims(self):
        xs = [float(j) for i in range(10) for j in [i] * (2 * i)]
        df = pd.DataFrame({
            'x': xs,
            'y': [v % 3 for v in range(len(xs))]
        })

        ds = Dataset(df)
        hist1 = histogram(
            ds,
            dimension='x',
            normed=False,
            num_bins=10,
            bin_range=[0, 10],
        )

        # Make sure hist1 dataset equal to original
        self.assertEqual(hist1.dataset, ds)

        # Check histogram data
        self.assertEqual(
            hist1.data,
            {'x': np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
             'x_count': np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])}
        )

        # Select histogram subset using the x and y dimensions
        hist2 = hist1.select(x=(4, None), y=2)

        # Check dataset down selection
        self.assertEqual(hist2.dataset, ds.select(x=(4, None), y=2))

        # Check histogram data. Bins should match and counts should be
        # reduced from hist1 due to selection
        self.assertEqual(
            hist2.data,
            {'x': np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
             'x_count': np.array([0, 0, 0, 0, 2, 4, 4, 4, 6, 6])}
        )

        # Check that selection using dim expression produces the same result
        hist3 = hist1.select((dim('x') >= 4) & (dim('y') == 2))
        self.assertEqual(hist3, hist2)


class DistributionTestCase(DatasetPropertyTestCase):

    def setUp(self):
        super(DistributionTestCase, self).setUp()
        self.distribution = self.ds.to(Distribution, kdims='a', groupby=[])

    def test_distribution_dataset(self):
        self.assertEqual(self.distribution.dataset, self.ds)
