import numpy as np
import pandas as pd
import pytest

from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.operation import function, histogram
from holoviews.testing import assert_element_equal

from ..utils import optional_dependencies

ds, ds_skip = optional_dependencies("datashader")
(dask, dd), dask_skip = optional_dependencies("dask", "dask.dataframe")
if ds:
    from holoviews.operation.datashader import datashade, dynspread, rasterize


class DatasetPropertyTestCase:

    def setup_method(self):
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
        assert ds is ds.dataset

        # Check pipeline
        ops = ds.pipeline.operations
        assert len(ops) == 1
        assert ops[0].output_type is Dataset
        assert_element_equal(ds, ds.pipeline(ds.dataset))

    def test_constructor_curve(self):
        element = Curve(self.df)
        expected = Dataset(
            self.df,
            kdims=self.df.columns[0],
            vdims=self.df.columns[1:].tolist(),
        )
        assert_element_equal(element.dataset, expected)

        # Check pipeline
        pipeline = element.pipeline
        assert len(pipeline.operations) == 1
        assert pipeline.operations[0].output_type is Curve
        assert_element_equal(element, element.pipeline(element.dataset))


class ToTestCase(DatasetPropertyTestCase):

    def test_to_element(self):
        curve = self.ds.to(Curve, 'a', 'b', groupby=[])
        curve2 = self.ds2.to(Curve, 'a', 'b', groupby=[])
        with pytest.raises(AssertionError):
            assert_element_equal(curve, curve2)

        assert_element_equal(curve.dataset, self.ds)

        scatter = curve.to(Scatter)
        assert_element_equal(scatter.dataset, self.ds)

        # Check pipeline
        ops = curve.pipeline.operations
        assert len(ops) == 2
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve

        # Execute pipeline
        assert_element_equal(curve.pipeline(curve.dataset), curve)
        assert_element_equal(
            curve.pipeline(self.ds2), curve2
        )

    def test_to_holomap(self):
        curve_hmap = self.ds.to(Curve, 'a', 'b', groupby=['c'])

        # Check HoloMap element datasets
        for v in self.df.c.drop_duplicates():
            curve = curve_hmap.data[(v,)]

            # check dataset
            assert_element_equal(
                curve.dataset, self.ds
            )

            # execute pipeline
            assert_element_equal(curve.pipeline(curve.dataset), curve)

    @dask_skip
    def test_to_holomap_dask(self):
        with dask.config.set({"dataframe.convert-string": False}):
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
            assert_element_equal(
                curve.dataset, self.ds
            )

            # Execute pipeline
            assert_element_equal(curve.pipeline(curve.dataset), curve)


class CloneTestCase(DatasetPropertyTestCase):
    def test_clone(self):
        # Dataset
        assert_element_equal(self.ds.clone().dataset, self.ds)

        # Curve
        curve = self.ds.to.curve('a', 'b', groupby=[])
        curve_clone = curve.clone()
        assert_element_equal(
            curve_clone.dataset,
            self.ds
        )

        # Check pipeline carried over
        assert curve.pipeline.operations == curve_clone.pipeline.operations[:2]

        # Execute pipeline
        assert_element_equal(curve.pipeline(curve.dataset), curve)

    def test_clone_new_data(self):
        # Replacing data during clone resets .dataset
        ds_clone = self.ds.clone(data=self.ds2.data)
        assert_element_equal(ds_clone.dataset, self.ds2)
        assert len(ds_clone.pipeline.operations) == 1

    def test_clone_dataset_kwarg_none(self):
        # Setting dataset=None prevents propagation of dataset to cloned object
        ds_clone = self.ds.clone(dataset=None)
        assert ds_clone is ds_clone.dataset


class ReindexTestCase(DatasetPropertyTestCase):
    def test_reindex_dataset(self):
        ds_ab = self.ds.reindex(kdims=['a'], vdims=['b'])
        ds2_ab = self.ds2.reindex(kdims=['a'], vdims=['b'])
        with pytest.raises(AssertionError):
            assert_element_equal(ds_ab, ds2_ab)

        assert_element_equal(ds_ab.dataset, self.ds)

        # Check pipeline
        ops = ds_ab.pipeline.operations
        assert len(ops) == 2
        assert ops[0].output_type is Dataset
        assert ops[1].method_name == 'reindex'
        assert ops[1].args == []
        assert ops[1].kwargs == dict(kdims=['a'], vdims=['b'])

        # Execute pipeline
        assert_element_equal(ds_ab.pipeline(ds_ab.dataset), ds_ab)
        assert_element_equal(
            ds_ab.pipeline(self.ds2), ds2_ab
        )

    def test_double_reindex_dataset(self):
        ds_ab = (self.ds
                 .reindex(kdims=['a'], vdims=['b', 'c'])
                 .reindex(kdims=['a'], vdims=['b']))
        ds2_ab = (self.ds2
                  .reindex(kdims=['a'], vdims=['b', 'c'])
                  .reindex(kdims=['a'], vdims=['b']))

        with pytest.raises(AssertionError):
            assert_element_equal(ds_ab, ds2_ab)

        assert_element_equal(ds_ab.dataset, self.ds)

        # Check pipeline
        ops = ds_ab.pipeline.operations
        assert len(ops) == 3
        assert ops[0].output_type is Dataset
        assert ops[1].method_name == 'reindex'
        assert ops[1].args == []
        assert ops[1].kwargs == dict(kdims=['a'], vdims=['b', 'c'])
        assert ops[2].method_name == 'reindex'
        assert ops[2].args == []
        assert ops[2].kwargs == dict(kdims=['a'], vdims=['b'])

        # Execute pipeline
        assert_element_equal(ds_ab.pipeline(ds_ab.dataset), ds_ab)
        assert_element_equal(ds_ab.pipeline(self.ds2), ds2_ab)

    def test_reindex_curve(self):
        curve_ba = self.ds.to(
            Curve, 'a', 'b', groupby=[]
        ).reindex(kdims='b', vdims='a')
        curve2_ba = self.ds2.to(
            Curve, 'a', 'b', groupby=[]
        ).reindex(kdims='b', vdims='a')

        with pytest.raises(AssertionError):
            assert_element_equal(curve_ba, curve2_ba)
        assert_element_equal(curve_ba.dataset, self.ds)

        # Check pipeline
        ops = curve_ba.pipeline.operations
        assert len(ops) == 3
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert ops[2].method_name == 'reindex'
        assert ops[2].args == []
        assert ops[2].kwargs == dict(kdims='b', vdims='a')

        # Execute pipeline
        assert_element_equal(curve_ba.pipeline(curve_ba.dataset), curve_ba)
        assert_element_equal(curve_ba.pipeline(self.ds2), curve2_ba)

    def test_double_reindex_curve(self):
        curve_ba = self.ds.to(
            Curve, 'a', ['b', 'c'], groupby=[]
        ).reindex(kdims='a', vdims='b').reindex(kdims='b', vdims='a')
        curve2_ba = self.ds2.to(
            Curve, 'a', ['b', 'c'], groupby=[]
        ).reindex(kdims='a', vdims='b').reindex(kdims='b', vdims='a')

        with pytest.raises(AssertionError):
            assert_element_equal(curve_ba, curve2_ba)

        assert_element_equal(curve_ba.dataset, self.ds)

        # Check pipeline
        ops = curve_ba.pipeline.operations
        assert len(ops) == 4
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert ops[2].method_name == 'reindex'
        assert ops[2].args == []
        assert ops[2].kwargs == dict(kdims='a', vdims='b')
        assert ops[3].method_name == 'reindex'
        assert ops[3].args == []
        assert ops[3].kwargs == dict(kdims='b', vdims='a')

        # Execute pipeline
        assert_element_equal(curve_ba.pipeline(curve_ba.dataset), curve_ba)
        assert_element_equal(curve_ba.pipeline(self.ds2), curve2_ba)


class IlocTestCase(DatasetPropertyTestCase):
    def test_iloc_dataset(self):
        ds_iloc = self.ds.iloc[[0, 2]]
        ds2_iloc = self.ds2.iloc[[0, 2]]
        with pytest.raises(AssertionError):
            assert_element_equal(ds_iloc, ds2_iloc)

        # Dataset
        assert_element_equal(ds_iloc.dataset, self.ds)

        # Check pipeline
        ops = ds_iloc.pipeline.operations
        assert len(ops) == 2
        assert ops[0].output_type is Dataset
        assert ops[1].method_name == '_perform_getitem'
        assert ops[1].args == [[0, 2]]
        assert ops[1].kwargs == {}

        # Execute pipeline
        assert_element_equal(ds_iloc.pipeline(ds_iloc.dataset), ds_iloc)
        assert_element_equal(ds_iloc.pipeline(self.ds2), ds2_iloc)

    def test_iloc_curve(self):
        # Curve
        curve_iloc = self.ds.to.curve('a', 'b', groupby=[]).iloc[[0, 2]]
        curve2_iloc = self.ds2.to.curve('a', 'b', groupby=[]).iloc[[0, 2]]

        with pytest.raises(AssertionError):
            assert_element_equal(curve_iloc, curve2_iloc)

        assert_element_equal(curve_iloc.dataset, self.ds)

        # Check pipeline
        ops = curve_iloc.pipeline.operations
        assert len(ops) == 3
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert ops[2].method_name == '_perform_getitem'
        assert ops[2].args == [[0, 2]]
        assert ops[2].kwargs == {}

        # Execute pipeline
        assert_element_equal(curve_iloc.pipeline(curve_iloc.dataset), curve_iloc)
        assert_element_equal(curve_iloc.pipeline(self.ds2), curve2_iloc)


class NdlocTestCase(DatasetPropertyTestCase):

    def setup_method(self):
        super().setup_method()
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

        with pytest.raises(AssertionError):
            assert_element_equal(ds_grid_ndloc, ds2_grid_ndloc)

        # Dataset
        assert_element_equal(
            ds_grid_ndloc.dataset,
            self.ds_grid
        )

        # Check pipeline
        ops = ds_grid_ndloc.pipeline.operations
        assert len(ops) == 2
        assert ops[0].output_type is Dataset
        assert ops[1].method_name == '_perform_getitem'
        assert ops[1].args == [(slice(0, 2, None), slice(1, 3, None),)]
        assert ops[1].kwargs == {}

        # Execute pipeline
        assert_element_equal(
            ds_grid_ndloc.pipeline(ds_grid_ndloc.dataset), ds_grid_ndloc
        )
        assert_element_equal(
            ds_grid_ndloc.pipeline(self.ds2_grid), ds2_grid_ndloc
        )


class SelectTestCase(DatasetPropertyTestCase):
    def test_select_dataset(self):
        ds_select = self.ds.select(b=10)
        ds2_select = self.ds2.select(b=10)

        with pytest.raises(AssertionError):
            assert_element_equal(ds_select, ds2_select)

        # Dataset
        assert_element_equal(ds_select.dataset, self.ds)

        # Check pipeline
        ops = ds_select.pipeline.operations
        assert len(ops) == 2
        assert ops[0].output_type is Dataset
        assert ops[1].method_name == 'select'
        assert ops[1].args == []
        assert ops[1].kwargs == {'b': 10}

        # Execute pipeline
        assert_element_equal(ds_select.pipeline(ds_select.dataset), ds_select)
        assert_element_equal(ds_select.pipeline(self.ds2), ds2_select)

    def test_select_curve(self):
        curve_select = self.ds.to.curve('a', 'b', groupby=[]).select(b=10)
        curve2_select = self.ds2.to.curve('a', 'b', groupby=[]).select(b=10)
        with pytest.raises(AssertionError):
            assert_element_equal(curve_select, curve2_select)

        # Curve
        assert_element_equal(curve_select.dataset, self.ds)

        # Check pipeline
        ops = curve_select.pipeline.operations
        assert len(ops) == 3
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert ops[2].method_name == 'select'
        assert ops[2].args == []
        assert ops[2].kwargs == {'b': 10}

        # Execute pipeline
        assert_element_equal(
            curve_select.pipeline(curve_select.dataset), curve_select
        )
        assert_element_equal(
            curve_select.pipeline(self.ds2), curve2_select
        )


class SortTestCase(DatasetPropertyTestCase):
    def test_sort_curve(self):
        curve_sorted = self.ds.to.curve('a', 'b', groupby=[]).sort('a')
        curve_sorted2 = self.ds2.to.curve('a', 'b', groupby=[]).sort('a')

        with pytest.raises(AssertionError):
            assert_element_equal(curve_sorted, curve_sorted2)

        # Curve
        assert_element_equal(curve_sorted.dataset, self.ds)

        # Check pipeline
        ops = curve_sorted.pipeline.operations
        assert len(ops) == 3
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert ops[2].method_name == 'sort'
        assert ops[2].args == ['a']
        assert ops[2].kwargs == {}

        # Execute pipeline
        assert_element_equal(
            curve_sorted.pipeline(curve_sorted.dataset), curve_sorted
        )
        assert_element_equal(
            curve_sorted.pipeline(self.ds2), curve_sorted2
        )


class SampleTestCase(DatasetPropertyTestCase):
    def test_sample_curve(self):
        curve_sampled = self.ds.to.curve('a', 'b', groupby=[]).sample([1, 2])
        curve_sampled2 = self.ds2.to.curve('a', 'b', groupby=[]).sample([1, 2])

        with pytest.raises(AssertionError):
            assert_element_equal(curve_sampled, curve_sampled2)

        # Curve
        assert_element_equal(curve_sampled.dataset, self.ds)

        # Check pipeline
        ops = curve_sampled.pipeline.operations
        assert len(ops) == 3
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert ops[2].method_name == 'sample'
        assert ops[2].args == [[1, 2]]
        assert ops[2].kwargs == {}

        # Execute pipeline
        assert_element_equal(
            curve_sampled.pipeline(curve_sampled.dataset), curve_sampled
        )
        assert_element_equal(
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

        with pytest.raises(AssertionError):
            assert_element_equal(ds_reduced, ds2_reduced)
        assert_element_equal(ds_reduced.dataset, self.ds)
        assert_element_equal(ds2_reduced.dataset, self.ds2)

        # Check pipeline
        ops = ds_reduced.pipeline.operations
        assert len(ops) == 3
        assert ops[0].output_type is Dataset
        assert ops[1].method_name == 'reindex'
        assert ops[2].method_name == 'reduce'
        assert ops[2].args == ['c']
        assert ops[2].kwargs == {'function': np.sum}

        # Execute pipeline
        assert_element_equal(ds_reduced.pipeline(ds_reduced.dataset), ds_reduced)
        assert_element_equal(ds_reduced.pipeline(self.ds2), ds2_reduced)


class AggregateTestCase(DatasetPropertyTestCase):
    def test_aggregate_dataset(self):
        ds_aggregated = self.ds.reindex(
            kdims=['b', 'c'], vdims=['a', 'd']
        ).aggregate('b', function=np.sum)

        ds2_aggregated = self.ds2.reindex(
            kdims=['b', 'c'], vdims=['a', 'd']
        ).aggregate('b', function=np.sum)

        with pytest.raises(AssertionError):
            assert_element_equal(ds_aggregated, ds2_aggregated)

        assert_element_equal(ds_aggregated.dataset, self.ds)
        assert_element_equal(ds2_aggregated.dataset, self.ds2)

        # Check pipeline
        ops = ds_aggregated.pipeline.operations
        assert len(ops) == 3
        assert ops[0].output_type is Dataset
        assert ops[1].method_name == 'reindex'
        assert ops[2].method_name == 'aggregate'
        assert ops[2].args == ['b']
        assert ops[2].kwargs == {'function': np.sum}

        # Execute pipeline
        assert_element_equal(
            ds_aggregated.pipeline(ds_aggregated.dataset), ds_aggregated
        )
        assert_element_equal(
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

        with pytest.raises(AssertionError):
            assert_element_equal(ds_groups, ds2_groups)

        for k in ds_groups.keys():
            ds_group = ds_groups[k]
            ds2_group = ds2_groups[k]

            # Check pipeline
            ops = ds_group.pipeline.operations
            assert len(ops) == 4
            assert ops[0].output_type is Dataset
            assert ops[1].method_name == 'reindex'
            assert ops[2].method_name == 'groupby'
            assert ops[2].args == ['b']
            assert ops[3].method_name == '__getitem__'
            assert ops[3].args == [k]

            # Execute pipeline
            assert_element_equal(ds_group.pipeline(ds_group.dataset), ds_group)
            assert_element_equal(ds_group.pipeline(self.ds2), ds2_group)


class AddDimensionTestCase(DatasetPropertyTestCase):
    def test_add_dimension_dataset(self):
        ds_dim_added = self.ds.add_dimension('new', 1, 17)
        ds2_dim_added = self.ds2.add_dimension('new', 1, 17)

        with pytest.raises(AssertionError):
            assert_element_equal(ds_dim_added, ds2_dim_added)

        # Check dataset
        assert_element_equal(ds_dim_added.dataset, self.ds)
        assert_element_equal(ds2_dim_added.dataset, self.ds2)

        # Check pipeline
        ops = ds_dim_added.pipeline.operations
        assert len(ops) == 2
        assert ops[0].output_type is Dataset
        assert ops[1].method_name == 'add_dimension'
        assert ops[1].args == ['new', 1, 17]
        assert ops[1].kwargs == {}

        # Execute pipeline
        assert_element_equal(
            ds_dim_added.pipeline(ds_dim_added.dataset), ds_dim_added
        )
        assert_element_equal(
            ds_dim_added.pipeline(self.ds2), ds2_dim_added,
        )


# Add execute pipeline test for each method, using a different dataset (ds2)
#
class HistogramTestCase(DatasetPropertyTestCase):

    def setup_method(self):
        super().setup_method()
        self.hist = self.ds.hist('a', adjoin=False, normed=False)

    def test_construction(self):
        assert_element_equal(self.hist.dataset, self.ds)

    def test_clone(self):
        assert_element_equal(self.hist.clone().dataset, self.ds)

    def test_select_single(self):
        sub_hist = self.hist.select(a=(1, None))
        assert_element_equal(sub_hist.dataset, self.ds)

        # Check pipeline
        ops = sub_hist.pipeline.operations
        assert len(ops) == 4
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Apply
        assert ops[2].method_name == '__call__'
        assert isinstance(ops[2].args[0], histogram)
        assert ops[3].method_name == 'select'
        assert ops[3].args == []
        assert ops[3].kwargs == {'a': (1, None)}

        # Execute pipeline
        assert_element_equal(sub_hist.pipeline(sub_hist.dataset), sub_hist)

    def test_select_multi(self):
        # Add second selection on b. b is a dimension in hist.dataset but
        # not in hist.  Make sure that we only apply the a selection (and not
        # the b selection) to the .dataset property
        sub_hist = self.hist.select(a=(1, None), b=100)

        with pytest.raises(AssertionError):
            assert_element_equal(
                sub_hist.dataset, self.ds.select(a=(1, None,), b=100)
            )

        # Check dataset unchanged
        assert_element_equal(sub_hist.dataset, self.ds)

        # Check pipeline
        ops = sub_hist.pipeline.operations
        assert len(ops) == 4
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Apply
        assert ops[2].method_name == '__call__'
        assert isinstance(ops[2].args[0], histogram)
        assert ops[3].method_name == 'select'
        assert ops[3].args == []
        assert ops[3].kwargs == {'a': (1, None), 'b': 100}

        # Execute pipeline
        assert_element_equal(sub_hist.pipeline(sub_hist.dataset), sub_hist)

    def test_hist_to_curve(self):
        # No exception thrown
        curve = self.hist.to.curve()

        # Check pipeline
        ops = curve.pipeline.operations
        assert len(ops) == 4
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Apply
        assert ops[2].method_name == '__call__'
        assert isinstance(ops[2].args[0], histogram)
        assert ops[3].output_type is Curve

        # Execute pipeline
        assert_element_equal(curve.pipeline(curve.dataset), curve)


class DistributionTestCase(DatasetPropertyTestCase):

    def setup_method(self):
        super().setup_method()
        self.distribution = self.ds.to(Distribution, kdims='a', groupby=[])

    def test_distribution_dataset(self):
        assert_element_equal(self.distribution.dataset, self.ds)

        # Execute pipeline
        assert_element_equal(
            self.distribution.pipeline(self.distribution.dataset),
            self.distribution,
        )


@ds_skip
class DatashaderTestCase(DatasetPropertyTestCase):

    def test_rasterize_curve(self):
        img = rasterize(
            self.ds.to(Curve, 'a', 'b', groupby=[]), dynamic=False
        )
        img2 = rasterize(
            self.ds2.to(Curve, 'a', 'b', groupby=[]), dynamic=False
        )
        with pytest.raises(AssertionError):
            assert_element_equal(img, img2)

        # Check dataset
        assert_element_equal(img.dataset, self.ds)

        # Check pipeline
        ops = img.pipeline.operations
        assert len(ops) == 3
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert isinstance(ops[2], rasterize)

        # Execute pipeline
        assert_element_equal(img.pipeline(img.dataset), img)
        assert_element_equal(img.pipeline(self.ds2), img2)

    def test_datashade_curve(self):
        rgb = dynspread(datashade(
            self.ds.to(Curve, 'a', 'b', groupby=[]), dynamic=False
        ), dynamic=False)
        rgb2 = dynspread(datashade(
            self.ds2.to(Curve, 'a', 'b', groupby=[]), dynamic=False
        ), dynamic=False)

        with pytest.raises(AssertionError):
            assert_element_equal(rgb, rgb2)

        # Check dataset
        assert_element_equal(rgb.dataset, self.ds)

        # Check pipeline
        ops = rgb.pipeline.operations
        assert len(ops) == 4
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert isinstance(ops[2], datashade)
        assert isinstance(ops[3], dynspread)

        # Execute pipeline
        assert_element_equal(rgb.pipeline(rgb.dataset), rgb)
        assert_element_equal(rgb.pipeline(self.ds2), rgb2)


class AccessorTestCase(DatasetPropertyTestCase):
    def test_apply_curve(self):
        curve = self.ds.to.curve('a', 'b', groupby=[]).apply(
            lambda c: Scatter(c.select(b=(20, None)).data)
        )
        curve2 = self.ds2.to.curve('a', 'b', groupby=[]).apply(
            lambda c: Scatter(c.select(b=(20, None)).data)
        )
        with pytest.raises(AssertionError):
            assert_element_equal(curve, curve2)

        # Check pipeline
        ops = curve.pipeline.operations
        assert len(ops) == 4
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert ops[2].output_type is Apply
        assert ops[2].kwargs == {'mode': None}
        assert ops[3].method_name == '__call__'

        # Execute pipeline
        assert_element_equal(curve.pipeline(curve.dataset), curve)
        assert_element_equal(curve.pipeline(self.ds2), curve2)

    def test_redim_curve(self):
        curve = self.ds.to.curve('a', 'b', groupby=[]).redim.unit(
            a='kg', b='m'
        )

        curve2 = self.ds2.to.curve('a', 'b', groupby=[]).redim.unit(
            a='kg', b='m'
        )
        with pytest.raises(AssertionError):
            assert_element_equal(curve, curve2)

        # Check pipeline
        ops = curve.pipeline.operations
        assert len(ops) == 4
        assert ops[0].output_type is Dataset
        assert ops[1].output_type is Curve
        assert ops[2].output_type is Redim
        assert ops[2].kwargs == {'mode': 'dataset'}
        assert ops[3].method_name == '__call__'

        # Execute pipeline
        assert_element_equal(curve.pipeline(curve.dataset), curve)
        assert_element_equal(curve.pipeline(self.ds2), curve2)


class OperationTestCase(DatasetPropertyTestCase):
    def test_propagate_dataset(self):
        op = function.instance(
            fn=lambda ds: ds.iloc[:5].clone(dataset=None, pipeline=None)
        )
        new_ds = op(self.ds)
        assert_element_equal(new_ds.dataset, self.ds)

    def test_do_not_propagate_dataset(self):
        op = function.instance(
            fn=lambda ds: ds.iloc[:5].clone(dataset=None, pipeline=None)
        )
        # Disable dataset propagation
        op._propagate_dataset = False
        new_ds = op(self.ds)
        assert_element_equal(new_ds.dataset, new_ds)
