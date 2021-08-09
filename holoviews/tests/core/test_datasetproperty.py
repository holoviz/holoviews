from unittest import SkipTest

import numpy as np
import pandas as pd

try:
    import dask.dataframe as dd
except:
    dd = None

from holoviews import Dataset, Curve, Dimension, Scatter, Distribution
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import histogram, function

try:
    from holoviews.operation.datashader import dynspread, datashade, rasterize
except:
    dynspread = datashade = rasterize = None



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

        self.ds2 = Dataset(
            self.df.iloc[2:],
            kdims=[
                Dimension('a', label="The a Column"),
                Dimension('b', label="The b Column"),
                Dimension('c', label="The c Column"),
                Dimension('d', label="The d Column"),
            ]
        )


class ConstructorTestCase(DatasetPropertyTestCase):
    def test_constructors_dataset(self):
        ds = Dataset(self.df)
        self.assertIs(ds, ds.dataset)

        # Check pipeline
        ops = ds.pipeline.operations
        self.assertEqual(len(ops), 1)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ds, ds.pipeline(ds.dataset))

    def test_constructor_curve(self):
        element = Curve(self.df)
        expected = Dataset(
            self.df,
            kdims=self.df.columns[0],
            vdims=self.df.columns[1:].tolist(),
        )
        self.assertEqual(element.dataset, expected)

        # Check pipeline
        pipeline = element.pipeline
        self.assertEqual(len(pipeline.operations), 1)
        self.assertIs(pipeline.operations[0].output_type, Curve)
        self.assertEqual(element, element.pipeline(element.dataset))


class ToTestCase(DatasetPropertyTestCase):

    def test_to_element(self):
        curve = self.ds.to(Curve, 'a', 'b', groupby=[])
        curve2 = self.ds2.to(Curve, 'a', 'b', groupby=[])
        self.assertNotEqual(curve, curve2)

        self.assertEqual(curve.dataset, self.ds)

        scatter = curve.to(Scatter)
        self.assertEqual(scatter.dataset, self.ds)

        # Check pipeline
        ops = curve.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)

        # Execute pipeline
        self.assertEqual(curve.pipeline(curve.dataset), curve)
        self.assertEqual(
            curve.pipeline(self.ds2), curve2
        )

    def test_to_holomap(self):
        curve_hmap = self.ds.to(Curve, 'a', 'b', groupby=['c'])

        # Check HoloMap element datasets
        for v in self.df.c.drop_duplicates():
            curve = curve_hmap.data[(v,)]

            # check dataset
            self.assertEqual(
                curve.dataset, self.ds
            )

            # execute pipeline
            self.assertEqual(curve.pipeline(curve.dataset), curve)

    def test_to_holomap_dask(self):
        if dd is None:
            raise SkipTest("Dask required to test .to with dask dataframe.")
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
                curve.dataset, self.ds
            )

            # Execute pipeline
            self.assertEqual(curve.pipeline(curve.dataset), curve)


class CloneTestCase(DatasetPropertyTestCase):
    def test_clone(self):
        # Dataset
        self.assertEqual(self.ds.clone().dataset, self.ds)

        # Curve
        curve = self.ds.to.curve('a', 'b', groupby=[])
        curve_clone = curve.clone()
        self.assertEqual(
            curve_clone.dataset,
            self.ds
        )

        # Check pipeline carried over
        self.assertEqual(
            curve.pipeline.operations, curve_clone.pipeline.operations[:2]
        )

        # Execute pipeline
        self.assertEqual(curve.pipeline(curve.dataset), curve)

    def test_clone_new_data(self):
        # Replacing data during clone resets .dataset
        ds_clone = self.ds.clone(data=self.ds2.data)
        self.assertEqual(ds_clone.dataset, self.ds2)
        self.assertEqual(len(ds_clone.pipeline.operations), 1)

    def test_clone_dataset_kwarg_none(self):
        # Setting dataset=None prevents propagation of dataset to cloned object
        ds_clone = self.ds.clone(dataset=None)
        self.assertIs(ds_clone, ds_clone.dataset)


class ReindexTestCase(DatasetPropertyTestCase):
    def test_reindex_dataset(self):
        ds_ab = self.ds.reindex(kdims=['a'], vdims=['b'])
        ds2_ab = self.ds2.reindex(kdims=['a'], vdims=['b'])
        self.assertNotEqual(ds_ab, ds2_ab)

        self.assertEqual(ds_ab.dataset, self.ds)

        # Check pipeline
        ops = ds_ab.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'reindex')
        self.assertEqual(ops[1].args, [])
        self.assertEqual(ops[1].kwargs, dict(kdims=['a'], vdims=['b']))

        # Execute pipeline
        self.assertEqual(ds_ab.pipeline(ds_ab.dataset), ds_ab)
        self.assertEqual(
            ds_ab.pipeline(self.ds2), ds2_ab
        )

    def test_double_reindex_dataset(self):
        ds_ab = (self.ds
                 .reindex(kdims=['a'], vdims=['b', 'c'])
                 .reindex(kdims=['a'], vdims=['b']))
        ds2_ab = (self.ds2
                  .reindex(kdims=['a'], vdims=['b', 'c'])
                  .reindex(kdims=['a'], vdims=['b']))
        self.assertNotEqual(ds_ab, ds2_ab)

        self.assertEqual(ds_ab.dataset, self.ds)

        # Check pipeline
        ops = ds_ab.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'reindex')
        self.assertEqual(ops[1].args, [])
        self.assertEqual(ops[1].kwargs, dict(kdims=['a'], vdims=['b', 'c']))
        self.assertEqual(ops[2].method_name, 'reindex')
        self.assertEqual(ops[2].args, [])
        self.assertEqual(ops[2].kwargs, dict(kdims=['a'], vdims=['b']))

        # Execute pipeline
        self.assertEqual(ds_ab.pipeline(ds_ab.dataset), ds_ab)
        self.assertEqual(
            ds_ab.pipeline(self.ds2), ds2_ab
        )

    def test_reindex_curve(self):
        curve_ba = self.ds.to(
            Curve, 'a', 'b', groupby=[]
        ).reindex(kdims='b', vdims='a')
        curve2_ba = self.ds2.to(
            Curve, 'a', 'b', groupby=[]
        ).reindex(kdims='b', vdims='a')
        self.assertNotEqual(curve_ba, curve2_ba)

        self.assertEqual(curve_ba.dataset, self.ds)

        # Check pipeline
        ops = curve_ba.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(ops[2].method_name, 'reindex')
        self.assertEqual(ops[2].args, [])
        self.assertEqual(ops[2].kwargs, dict(kdims='b', vdims='a'))

        # Execute pipeline
        self.assertEqual(curve_ba.pipeline(curve_ba.dataset), curve_ba)
        self.assertEqual(
            curve_ba.pipeline(self.ds2), curve2_ba
        )

    def test_double_reindex_curve(self):
        curve_ba = self.ds.to(
            Curve, 'a', ['b', 'c'], groupby=[]
        ).reindex(kdims='a', vdims='b').reindex(kdims='b', vdims='a')
        curve2_ba = self.ds2.to(
            Curve, 'a', ['b', 'c'], groupby=[]
        ).reindex(kdims='a', vdims='b').reindex(kdims='b', vdims='a')
        self.assertNotEqual(curve_ba, curve2_ba)

        self.assertEqual(curve_ba.dataset, self.ds)

        # Check pipeline
        ops = curve_ba.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(ops[2].method_name, 'reindex')
        self.assertEqual(ops[2].args, [])
        self.assertEqual(ops[2].kwargs, dict(kdims='a', vdims='b'))
        self.assertEqual(ops[3].method_name, 'reindex')
        self.assertEqual(ops[3].args, [])
        self.assertEqual(ops[3].kwargs, dict(kdims='b', vdims='a'))

        # Execute pipeline
        self.assertEqual(curve_ba.pipeline(curve_ba.dataset), curve_ba)
        self.assertEqual(
            curve_ba.pipeline(self.ds2), curve2_ba
        )


class IlocTestCase(DatasetPropertyTestCase):
    def test_iloc_dataset(self):
        ds_iloc = self.ds.iloc[[0, 2]]
        ds2_iloc = self.ds2.iloc[[0, 2]]
        self.assertNotEqual(ds_iloc, ds2_iloc)

        # Dataset
        self.assertEqual(
            ds_iloc.dataset,
            self.ds
        )

        # Check pipeline
        ops = ds_iloc.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, '_perform_getitem')
        self.assertEqual(ops[1].args, [[0, 2]])
        self.assertEqual(ops[1].kwargs, {})

        # Execute pipeline
        self.assertEqual(ds_iloc.pipeline(ds_iloc.dataset), ds_iloc)
        self.assertEqual(
            ds_iloc.pipeline(self.ds2), ds2_iloc
        )

    def test_iloc_curve(self):
        # Curve
        curve_iloc = self.ds.to.curve('a', 'b', groupby=[]).iloc[[0, 2]]
        curve2_iloc = self.ds2.to.curve('a', 'b', groupby=[]).iloc[[0, 2]]
        self.assertNotEqual(curve_iloc, curve2_iloc)

        self.assertEqual(
            curve_iloc.dataset,
            self.ds
        )

        # Check pipeline
        ops = curve_iloc.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(ops[2].method_name, '_perform_getitem')
        self.assertEqual(ops[2].args, [[0, 2]])
        self.assertEqual(ops[2].kwargs, {})

        # Execute pipeline
        self.assertEqual(curve_iloc.pipeline(curve_iloc.dataset), curve_iloc)
        self.assertEqual(
            curve_iloc.pipeline(self.ds2), curve2_iloc
        )


class NdlocTestCase(DatasetPropertyTestCase):

    def setUp(self):
        super().setUp()
        self.ds_grid = Dataset(
            (np.arange(4),
             np.arange(3),
             np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])),
            kdims=['x', 'y'],
            vdims='z'
        )

        self.ds2_grid = Dataset(
            (np.arange(3),
             np.arange(3),
             np.array([[1, 2, 4],
                       [5, 6, 8],
                       [9, 10, 12]])),
            kdims=['x', 'y'],
            vdims='z'
        )

    def test_ndloc_dataset(self):
        ds_grid_ndloc = self.ds_grid.ndloc[0:2, 1:3]
        ds2_grid_ndloc = self.ds2_grid.ndloc[0:2, 1:3]
        self.assertNotEqual(ds_grid_ndloc, ds2_grid_ndloc)

        # Dataset
        self.assertEqual(
            ds_grid_ndloc.dataset,
            self.ds_grid
        )

        # Check pipeline
        ops = ds_grid_ndloc.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, '_perform_getitem')
        self.assertEqual(
            ops[1].args, [(slice(0, 2, None), slice(1, 3, None))]
        )
        self.assertEqual(ops[1].kwargs, {})

        # Execute pipeline
        self.assertEqual(
            ds_grid_ndloc.pipeline(ds_grid_ndloc.dataset), ds_grid_ndloc
        )
        self.assertEqual(
            ds_grid_ndloc.pipeline(self.ds2_grid), ds2_grid_ndloc
        )


class SelectTestCase(DatasetPropertyTestCase):
    def test_select_dataset(self):
        ds_select = self.ds.select(b=10)
        ds2_select = self.ds2.select(b=10)
        self.assertNotEqual(ds_select, ds2_select)


        # Dataset
        self.assertEqual(
            ds_select.dataset,
            self.ds
        )

        # Check pipeline
        ops = ds_select.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'select')
        self.assertEqual(ops[1].args, [])
        self.assertEqual(ops[1].kwargs, {'b': 10})

        # Execute pipeline
        self.assertEqual(ds_select.pipeline(ds_select.dataset), ds_select)
        self.assertEqual(
            ds_select.pipeline(self.ds2), ds2_select
        )

    def test_select_curve(self):
        curve_select = self.ds.to.curve('a', 'b', groupby=[]).select(b=10)
        curve2_select = self.ds2.to.curve('a', 'b', groupby=[]).select(b=10)
        self.assertNotEqual(curve_select, curve2_select)

        # Curve
        self.assertEqual(
            curve_select.dataset,
            self.ds
        )

        # Check pipeline
        ops = curve_select.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(ops[2].method_name, 'select')
        self.assertEqual(ops[2].args, [])
        self.assertEqual(ops[2].kwargs, {'b': 10})

        # Execute pipeline
        self.assertEqual(
            curve_select.pipeline(curve_select.dataset), curve_select
        )
        self.assertEqual(
            curve_select.pipeline(self.ds2), curve2_select
        )


class SortTestCase(DatasetPropertyTestCase):
    def test_sort_curve(self):
        curve_sorted = self.ds.to.curve('a', 'b', groupby=[]).sort('a')
        curve_sorted2 = self.ds2.to.curve('a', 'b', groupby=[]).sort('a')
        self.assertNotEqual(curve_sorted, curve_sorted2)

        # Curve
        self.assertEqual(
            curve_sorted.dataset,
            self.ds
        )

        # Check pipeline
        ops = curve_sorted.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(ops[2].method_name, 'sort')
        self.assertEqual(ops[2].args, ['a'])
        self.assertEqual(ops[2].kwargs, {})

        # Execute pipeline
        self.assertEqual(
            curve_sorted.pipeline(curve_sorted.dataset), curve_sorted
        )
        self.assertEqual(
            curve_sorted.pipeline(self.ds2), curve_sorted2
        )


class SampleTestCase(DatasetPropertyTestCase):
    def test_sample_curve(self):
        curve_sampled = self.ds.to.curve('a', 'b', groupby=[]).sample([1, 2])
        curve_sampled2 = self.ds2.to.curve('a', 'b', groupby=[]).sample([1, 2])
        self.assertNotEqual(curve_sampled, curve_sampled2)

        # Curve
        self.assertEqual(
            curve_sampled.dataset,
            self.ds
        )

        # Check pipeline
        ops = curve_sampled.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(ops[2].method_name, 'sample')
        self.assertEqual(ops[2].args, [[1, 2]])
        self.assertEqual(ops[2].kwargs, {})

        # Execute pipeline
        self.assertEqual(
            curve_sampled.pipeline(curve_sampled.dataset), curve_sampled
        )
        self.assertEqual(
            curve_sampled.pipeline(self.ds2), curve_sampled2
        )


class ReduceTestCase(DatasetPropertyTestCase):
    def test_reduce_dataset(self):
        ds_reduced = self.ds.reindex(
            kdims=['b', 'c'], vdims=['a', 'd']
        ).reduce('c', function=np.sum)

        ds2_reduced = self.ds2.reindex(
            kdims=['b', 'c'], vdims=['a', 'd']
        ).reduce('c', function=np.sum)

        self.assertNotEqual(ds_reduced, ds2_reduced)
        self.assertEqual(ds_reduced.dataset, self.ds)
        self.assertEqual(ds2_reduced.dataset, self.ds2)

        # Check pipeline
        ops = ds_reduced.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'reindex')
        self.assertEqual(ops[2].method_name, 'reduce')
        self.assertEqual(ops[2].args, ['c'])
        self.assertEqual(ops[2].kwargs, {'function': np.sum})

        # Execute pipeline
        self.assertEqual(ds_reduced.pipeline(ds_reduced.dataset), ds_reduced)
        self.assertEqual(
            ds_reduced.pipeline(self.ds2), ds2_reduced
        )


class AggregateTestCase(DatasetPropertyTestCase):
    def test_aggregate_dataset(self):
        ds_aggregated = self.ds.reindex(
            kdims=['b', 'c'], vdims=['a', 'd']
        ).aggregate('b', function=np.sum)

        ds2_aggregated = self.ds2.reindex(
            kdims=['b', 'c'], vdims=['a', 'd']
        ).aggregate('b', function=np.sum)

        self.assertNotEqual(ds_aggregated, ds2_aggregated)
        self.assertEqual(ds_aggregated.dataset, self.ds)
        self.assertEqual(ds2_aggregated.dataset, self.ds2)

        # Check pipeline
        ops = ds_aggregated.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'reindex')
        self.assertEqual(ops[2].method_name, 'aggregate')
        self.assertEqual(ops[2].args, ['b'])
        self.assertEqual(ops[2].kwargs, {'function': np.sum})

        # Execute pipeline
        self.assertEqual(
            ds_aggregated.pipeline(ds_aggregated.dataset), ds_aggregated
        )
        self.assertEqual(
            ds_aggregated.pipeline(self.ds2), ds2_aggregated
        )


class GroupbyTestCase(DatasetPropertyTestCase):
    def test_groupby_dataset(self):
        ds_groups = self.ds.reindex(
            kdims=['b', 'c'], vdims=['a', 'd']
        ).groupby('b')

        ds2_groups = self.ds2.reindex(
            kdims=['b', 'c'], vdims=['a', 'd']
        ).groupby('b')

        self.assertNotEqual(ds_groups, ds2_groups)
        for k in ds_groups.keys():
            ds_group = ds_groups[k]
            ds2_group = ds2_groups[k]

            # Check pipeline
            ops = ds_group.pipeline.operations
            self.assertNotEqual(len(ops), 3)
            self.assertIs(ops[0].output_type, Dataset)
            self.assertEqual(ops[1].method_name, 'reindex')
            self.assertEqual(ops[2].method_name, 'groupby')
            self.assertEqual(ops[2].args, ['b'])
            self.assertEqual(ops[3].method_name, '__getitem__')
            self.assertEqual(ops[3].args, [k])

            # Execute pipeline
            self.assertEqual(ds_group.pipeline(ds_group.dataset), ds_group)
            self.assertEqual(
                ds_group.pipeline(self.ds2), ds2_group
            )


class AddDimensionTestCase(DatasetPropertyTestCase):
    def test_add_dimension_dataset(self):
        ds_dim_added = self.ds.add_dimension('new', 1, 17)
        ds2_dim_added = self.ds2.add_dimension('new', 1, 17)
        self.assertNotEqual(ds_dim_added, ds2_dim_added)

        # Check dataset
        self.assertEqual(ds_dim_added.dataset, self.ds)
        self.assertEqual(ds2_dim_added.dataset, self.ds2)

        # Check pipeline
        ops = ds_dim_added.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'add_dimension')
        self.assertEqual(ops[1].args, ['new', 1, 17])
        self.assertEqual(ops[1].kwargs, {})

        # Execute pipeline
        self.assertEqual(
            ds_dim_added.pipeline(ds_dim_added.dataset), ds_dim_added
        )
        self.assertEqual(
            ds_dim_added.pipeline(self.ds2), ds2_dim_added,
        )


# Add execute pipeline test for each method, using a different dataset (ds2)
#
class HistogramTestCase(DatasetPropertyTestCase):

    def setUp(self):
        super().setUp()
        self.hist = self.ds.hist('a', adjoin=False, normed=False)

    def test_construction(self):
        self.assertEqual(self.hist.dataset, self.ds)

    def test_clone(self):
        self.assertEqual(self.hist.clone().dataset, self.ds)

    def test_select_single(self):
        sub_hist = self.hist.select(a=(1, None))
        self.assertEqual(sub_hist.dataset, self.ds)

        # Check pipeline
        ops = sub_hist.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Apply)
        self.assertEqual(ops[2].method_name, '__call__')
        self.assertIsInstance(ops[2].args[0], histogram)
        self.assertEqual(ops[3].method_name, 'select')
        self.assertEqual(ops[3].args, [])
        self.assertEqual(ops[3].kwargs, {'a': (1, None)})

        # Execute pipeline
        self.assertEqual(sub_hist.pipeline(sub_hist.dataset), sub_hist)

    def test_select_multi(self):
        # Add second selection on b. b is a dimension in hist.dataset but
        # not in hist.  Make sure that we only apply the a selection (and not
        # the b selection) to the .dataset property
        sub_hist = self.hist.select(a=(1, None), b=100)

        self.assertNotEqual(
            sub_hist.dataset,
            self.ds.select(a=(1, None), b=100)
        )

        # Check dataset unchanged
        self.assertEqual(
            sub_hist.dataset,
            self.ds
        )

        # Check pipeline
        ops = sub_hist.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Apply)
        self.assertEqual(ops[2].method_name, '__call__')
        self.assertIsInstance(ops[2].args[0], histogram)
        self.assertEqual(ops[3].method_name, 'select')
        self.assertEqual(ops[3].args, [])
        self.assertEqual(ops[3].kwargs, {'a': (1, None), 'b': 100})

        # Execute pipeline
        self.assertEqual(sub_hist.pipeline(sub_hist.dataset), sub_hist)

    def test_hist_to_curve(self):
        # No exception thrown
        curve = self.hist.to.curve()

        # Check pipeline
        ops = curve.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Apply)
        self.assertEqual(ops[2].method_name, '__call__')
        self.assertIsInstance(ops[2].args[0], histogram)
        self.assertIs(ops[3].output_type, Curve)

        # Execute pipeline
        self.assertEqual(curve.pipeline(curve.dataset), curve)


class DistributionTestCase(DatasetPropertyTestCase):

    def setUp(self):
        super().setUp()
        self.distribution = self.ds.to(Distribution, kdims='a', groupby=[])

    def test_distribution_dataset(self):
        self.assertEqual(self.distribution.dataset, self.ds)

        # Execute pipeline
        self.assertEqual(
            self.distribution.pipeline(self.distribution.dataset),
            self.distribution,
        )


class DatashaderTestCase(DatasetPropertyTestCase):

    def setUp(self):
        if None in (rasterize, datashade, dynspread):
            raise SkipTest('Datashader could not be imported and cannot be tested.')
        super().setUp()

    def test_rasterize_curve(self):
        img = rasterize(
            self.ds.to(Curve, 'a', 'b', groupby=[]), dynamic=False
        )
        img2 = rasterize(
            self.ds2.to(Curve, 'a', 'b', groupby=[]), dynamic=False
        )
        self.assertNotEqual(img, img2)

        # Check dataset
        self.assertEqual(img.dataset, self.ds)

        # Check pipeline
        ops = img.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertIsInstance(ops[2], rasterize)

        # Execute pipeline
        self.assertEqual(img.pipeline(img.dataset), img)
        self.assertEqual(img.pipeline(self.ds2), img2)

    def test_datashade_curve(self):
        rgb = dynspread(datashade(
            self.ds.to(Curve, 'a', 'b', groupby=[]), dynamic=False
        ), dynamic=False)
        rgb2 = dynspread(datashade(
            self.ds2.to(Curve, 'a', 'b', groupby=[]), dynamic=False
        ), dynamic=False)
        self.assertNotEqual(rgb, rgb2)

        # Check dataset
        self.assertEqual(rgb.dataset, self.ds)

        # Check pipeline
        ops = rgb.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertIsInstance(ops[2], datashade)
        self.assertIsInstance(ops[3], dynspread)

        # Execute pipeline
        self.assertEqual(rgb.pipeline(rgb.dataset), rgb)
        self.assertEqual(rgb.pipeline(self.ds2), rgb2)


class AccessorTestCase(DatasetPropertyTestCase):
    def test_apply_curve(self):
        curve = self.ds.to.curve('a', 'b', groupby=[]).apply(
            lambda c: Scatter(c.select(b=(20, None)).data)
        )
        curve2 = self.ds2.to.curve('a', 'b', groupby=[]).apply(
            lambda c: Scatter(c.select(b=(20, None)).data)
        )
        self.assertNotEqual(curve, curve2)

        # Check pipeline
        ops = curve.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertIs(ops[2].output_type, Apply)
        self.assertEqual(ops[2].kwargs, {'mode': None})
        self.assertEqual(ops[3].method_name, '__call__')

        # Execute pipeline
        self.assertEqual(curve.pipeline(curve.dataset), curve)
        self.assertEqual(
            curve.pipeline(self.ds2), curve2
        )

    def test_redim_curve(self):
        curve = self.ds.to.curve('a', 'b', groupby=[]).redim.unit(
            a='kg', b='m'
        )

        curve2 = self.ds2.to.curve('a', 'b', groupby=[]).redim.unit(
            a='kg', b='m'
        )
        self.assertNotEqual(curve, curve2)

        # Check pipeline
        ops = curve.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertIs(ops[2].output_type, Redim)
        self.assertEqual(ops[2].kwargs, {'mode': 'dataset'})
        self.assertEqual(ops[3].method_name, '__call__')

        # Execute pipeline
        self.assertEqual(curve.pipeline(curve.dataset), curve)
        self.assertEqual(
            curve.pipeline(self.ds2), curve2
        )


class OperationTestCase(DatasetPropertyTestCase):
    def test_propagate_dataset(self):
        op = function.instance(
            fn=lambda ds: ds.iloc[:5].clone(dataset=None, pipeline=None)
        )
        new_ds = op(self.ds)
        self.assertEqual(new_ds.dataset, self.ds)

    def test_do_not_propagate_dataset(self):
        op = function.instance(
            fn=lambda ds: ds.iloc[:5].clone(dataset=None, pipeline=None)
        )
        # Disable dataset propagation
        op._propagate_dataset = False
        new_ds = op(self.ds)
        self.assertEqual(new_ds.dataset, new_ds)
