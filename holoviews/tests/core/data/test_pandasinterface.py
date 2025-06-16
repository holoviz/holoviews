import numpy as np
import pandas as pd
import pytest

from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.data.pandas import PandasInterface
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter

from .base import HeterogeneousColumnTests, InterfaceTests


class BasePandasInterfaceTests(HeterogeneousColumnTests, InterfaceTests):
    """
    Test for the PandasInterface.
    """

    __test__ = False

    def test_duplicate_dimension_constructor(self):
        ds = Dataset(([1, 2, 3], [1, 2, 3]), ['A', 'B'], ['A'])
        self.assertEqual(list(ds.data.columns), ['A', 'B'])

    def test_dataset_empty_list_init_dtypes(self):
        dataset = Dataset([], kdims=['x'], vdims=['y'])
        for d in 'xy':
            self.assertEqual(dataset.dimension_values(d).dtype, np.float64)

    def test_dataset_series_construct(self):
        ds = Scatter(pd.Series([1, 2, 3], name='A'))
        self.assertEqual(ds, Scatter(([0, 1, 2], [1, 2, 3]), 'index', 'A'))

    def test_dataset_df_construct_autoindex(self):
        ds = Scatter(pd.DataFrame([1, 2, 3], columns=['A'], index=[1, 2, 3]), 'test', 'A')
        self.assertEqual(ds, Scatter(([0, 1, 2], [1, 2, 3]), 'test', 'A'))

    def test_dataset_df_construct_not_autoindex(self):
        ds = Scatter(pd.DataFrame([1, 2, 3], columns=['A'], index=[1, 2, 3]), 'index', 'A')
        self.assertEqual(ds, Scatter(([1, 2, 3], [1, 2, 3]), 'index', 'A'))

    def test_dataset_single_column_construct(self):
        ds = Scatter(pd.DataFrame([1, 2, 3], columns=['A']))
        self.assertEqual(ds, Scatter(([0, 1, 2], [1, 2, 3]), 'index', 'A'))

    def test_dataset_df_duplicate_columns_raises(self):
        df = pd.DataFrame(np.random.randint(-100,100, size=(100, 2)), columns=list("AB"))
        with self.assertRaises(DataError):
            Dataset(df[['A', 'A']])

    def test_dataset_extract_vdims(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Dataset(df, kdims=['x'])
        self.assertEqual(ds.vdims, [Dimension('y'), Dimension('z')])

    def test_dataset_process_index(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Dataset(df, 'index')
        self.assertEqual(ds.kdims, [Dimension('index')])
        self.assertEqual(ds.vdims, [Dimension('x'), Dimension('y'), Dimension('z')])

    def test_dataset_extract_kdims_and_vdims_no_bounds(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Dataset(df)
        self.assertEqual(ds.kdims, [Dimension('x'), Dimension('y'), Dimension('z')])
        self.assertEqual(ds.vdims, [])

    def test_dataset_extract_kdims(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Distribution(df)
        self.assertEqual(ds.kdims, [Dimension('x')])

    def test_dataset_extract_kdims_and_vdims(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Points(df)
        self.assertEqual(ds.kdims, [Dimension('x'), Dimension('y')])
        self.assertEqual(ds.vdims, [Dimension('z')])

    def test_dataset_element_allowing_two_kdims_with_one_default_kdim(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Scatter(df)
        self.assertEqual(ds.kdims, [Dimension('x')])
        self.assertEqual(ds.vdims, [Dimension('y'), Dimension('z')])

    def test_dataset_extract_kdims_with_vdims_defined(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Points(df, vdims=['x'])
        self.assertEqual(ds.kdims, [Dimension('y'), Dimension('z')])
        self.assertEqual(ds.vdims, [Dimension('x')])

    def test_dataset_extract_all_kdims_with_vdims_defined(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Dataset(df, vdims=['x'])
        self.assertEqual(ds.kdims, [Dimension('y'), Dimension('z')])
        self.assertEqual(ds.vdims, [Dimension('x')])

    def test_dataset_extract_kdims_declare_no_vdims(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Points(df, vdims=[])
        self.assertEqual(ds.kdims, [Dimension('x'), Dimension('y')])
        self.assertEqual(ds.vdims, [])

    def test_dataset_extract_no_kdims_extract_only_vdims(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Dataset(df, kdims=[])
        self.assertEqual(ds.kdims, [])
        self.assertEqual(ds.vdims, [Dimension('x'), Dimension('y'), Dimension('z')])

    def test_dataset_extract_vdims_with_kdims_defined(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]},
                          columns=['x', 'y', 'z'])
        ds = Points(df, kdims=['x', 'z'])
        self.assertEqual(ds.kdims, [Dimension('x'), Dimension('z')])
        self.assertEqual(ds.vdims, [Dimension('y')])

    def test_multi_dimension_groupby(self):
        x, y, z = list('AB'*10), np.arange(20)%3, np.arange(20)
        ds = Dataset((x, y, z), kdims=['x', 'y'], vdims=['z'],  datatype=[self.datatype])
        keys = [('A', 0), ('B', 1), ('A', 2), ('B', 0), ('A', 1), ('B', 2)]
        grouped = ds.groupby(['x', 'y'])
        self.assertEqual(grouped.keys(), keys)
        group = Dataset({'z': [5, 11, 17]}, vdims=['z'])
        self.assertEqual(grouped.last, group)

    def test_dataset_simple_dict_sorted(self):
        dataset = Dataset({2: 2, 1: 1, 3: 3}, kdims=['x'], vdims=['y'])
        self.assertEqual(dataset, Dataset([(i, i) for i in range(1, 4)],
                                          kdims=['x'], vdims=['y']))

    def test_dataset_conversion_with_index(self):
        df = pd.DataFrame({'y': [1, 2, 3]}, index=[0, 1, 2])
        scatter = Dataset(df).to(Scatter, 'index', 'y')
        self.assertEqual(scatter, Scatter(([0, 1, 2], [1, 2, 3]), 'index', 'y'))

    def test_dataset_conversion_groupby_with_index(self):
        df = pd.DataFrame({'y': [1, 2, 3], 'x': [0, 0, 1]}, index=[0, 1, 2])
        scatters = Dataset(df).to(Scatter, 'index', 'y')
        hmap = HoloMap({0: Scatter(([0, 1], [1, 2]), 'index', 'y'),
                        1: Scatter([(2, 3)], 'index', 'y')}, 'x')
        self.assertEqual(scatters, hmap)

    def test_dataset_from_multi_index(self):
        df = pd.DataFrame({'x': np.arange(10), 'y': np.arange(10), 'z': np.random.rand(10)})
        ds = Dataset(df.groupby(['x', 'y']).mean(), ['x', 'y'])
        self.assertEqual(ds, Dataset(df, ['x', 'y']))

    def test_dataset_from_multi_index_tuple_dims(self):
        df = pd.DataFrame({'x': np.arange(10), 'y': np.arange(10), 'z': np.random.rand(10)})
        ds = Dataset(df.groupby(['x', 'y']).mean(), [('x', 'X'), ('y', 'Y')])
        self.assertEqual(ds, Dataset(df, [('x', 'X'), ('y', 'Y')]))

    def test_dataset_with_interface_column(self):
        df = pd.DataFrame([1], columns=['interface'])
        ds = Dataset(df)
        self.assertEqual(list(ds.data.columns), ['interface'])

    def test_dataset_range_with_object_index(self):
        df = pd.DataFrame(range(4), columns=["values"], index=list("BADC"))
        ds = Dataset(df, kdims='index')
        assert ds.range('index') == ('A', 'D')


class PandasInterfaceTests(BasePandasInterfaceTests):

    datatype = 'dataframe'
    data_type = pd.DataFrame

    __test__ = True

    def test_data_with_tz(self):
        dates = pd.date_range("2018-01-01", periods=3, freq="h")
        dates_tz = dates.tz_localize("UTC")
        df = pd.DataFrame({"dates": dates_tz})
        data = Dataset(df).dimension_values("dates")
        np.testing.assert_equal(dates, data)

    def test_data_groupby_categorial(self):
        # Test for https://github.com/holoviz/holoviews/issues/6305
        df = pd.DataFrame({"y": [1, 2], "by": ["A", "B"]})
        df["by"] = pd.Categorical(df["by"])
        ds = Dataset(df, kdims="index", vdims="y").to(Scatter, groupby="by")
        assert ds.keys() == ["A", "B"]

    @pytest.mark.xfail(reason="Breaks hvplot")
    def test_reindex(self):
        ds = Dataset(pd.DataFrame({'x': np.arange(10), 'y': np.arange(10), 'z': np.random.rand(10)}))
        df = ds.interface.reindex(ds, ['x'])
        assert df.index.names == ['x']
        df = ds.interface.reindex(ds, ['y'])
        assert df.index.names == ['y']


class PandasInterfaceMultiIndex(HeterogeneousColumnTests, InterfaceTests):
    datatype = 'dataframe'
    data_type = pd.DataFrame

    __test__ = True

    def setUp(self):
        frame = pd.DataFrame({"number": [1, 1, 2, 2], "color": ["red", "blue", "red", "blue"]})
        index = pd.MultiIndex.from_frame(frame, names=("number", "color"))
        self.df = pd.DataFrame(range(4), index=index, columns=["values"])
        super().setUp()

    def test_lexsort_depth_import(self):
        # Indexing relies on knowing the lexsort_depth but this is a
        # private import so we want to know should this import ever
        # be changed
        from pandas.core.indexes.multi import _lexsort_depth  # noqa

    def test_no_kdims(self):
        ds = Dataset(self.df)
        assert ds.kdims == [Dimension("values")]
        assert isinstance(ds.data.index, pd.MultiIndex)

    def test_index_kdims(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        assert ds.kdims == [Dimension("number"), Dimension("color")]
        assert ds.vdims == [Dimension("values")]
        assert isinstance(ds.data.index, pd.MultiIndex)

    def test_index_aggregate(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        expected = pd.DataFrame({'number': [1, 2], 'values': [0.5, 2.5], 'values_var': [0.25, 0.25]})
        agg = ds.aggregate("number", function=np.mean, spreadfn=np.var)
        pd.testing.assert_frame_equal(agg.data, expected)

    def test_index_select_monotonic(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.select(number=1)
        expected = pd.DataFrame({'color': ['red', 'blue'], 'values': [0, 1], 'number': [1, 1]}).set_index(['number', 'color'])
        assert isinstance(selected.data.index, pd.MultiIndex)
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_index_select(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.select(number=1)
        expected = pd.DataFrame({'color': ['red', 'blue'], 'values': [0, 1], 'number': [1, 1]}).set_index(['number', 'color'])
        assert isinstance(selected.data.index, pd.MultiIndex)
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_index_select_all_indexes(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.select(number=1, color='red')
        assert selected == 0

    def test_index_select_all_indexes_lists(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.select(number=[1], color=['red'])
        expected = pd.DataFrame({'color': ['red'], 'values': [0], 'number': [1]}).set_index(['number', 'color'])
        assert isinstance(selected.data.index, pd.MultiIndex)
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_index_select_all_indexes_slice_and_scalar(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.select(number=(0, 1), color='red')
        expected = pd.DataFrame({'color': ['red'], 'values': [0], 'number': [1]}).set_index(['number', 'color'])
        assert isinstance(selected.data.index, pd.MultiIndex)
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_iloc_scalar_scalar_only_index(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.iloc[0, 0]
        expected = 1
        assert selected == expected

    def test_iloc_slice_scalar_only_index(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.iloc[:, 0]
        expected = self.df.reset_index()[["number"]]
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_iloc_slice_slice_only_index(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.iloc[:, :2]
        expected = self.df.reset_index()[["number", "color"]]
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_iloc_scalar_slice_only_index(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.iloc[0, :2]
        expected = pd.DataFrame({"number": 1, "color": "red"}, index=[0])
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_iloc_scalar_scalar(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.iloc[0, 2]
        expected = 0
        assert selected == expected

    def test_iloc_slice_scalar(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.iloc[:, 2]
        expected = self.df.iloc[:, [0]]
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_iloc_slice_slice(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.iloc[:, :3]
        expected = self.df.iloc[:, [0]]
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_iloc_scalar_slice(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.iloc[0, :3]
        expected = self.df.iloc[[0], [0]]
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_out_of_bounds(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        with pytest.raises(ValueError, match="column is out of bounds"):
            ds.iloc[0, 3]

    def test_sort(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        sorted_ds = ds.sort("color")
        np.testing.assert_array_equal(sorted_ds.dimension_values("values"), [1, 3, 0, 2])
        np.testing.assert_array_equal(sorted_ds.dimension_values("number"), [1, 2, 1, 2])

    def test_select_monotonic(self):
        ds = Dataset(self.df.sort_index(), kdims=["number", "color"])
        selected = ds.select(color="red")
        pd.testing.assert_frame_equal(selected.data, self.df.iloc[[0, 2], :])

        selected = ds.select(number=1, color='red')
        assert selected == 0

    def test_select_not_monotonic(self):
        frame = pd.DataFrame({"number": [1, 1, 2, 2], "color": [2, 1, 2, 1]})
        index = pd.MultiIndex.from_frame(frame, names=frame.columns)
        df = pd.DataFrame(range(4), index=index, columns=["values"])
        ds = Dataset(df, kdims=list(frame.columns))

        data = ds.select(color=slice(2, 3)).data
        expected = pd.DataFrame({"number": [1, 2], "color": [2, 2], "values": [0, 2]}).set_index(['number', 'color'])
        pd.testing.assert_frame_equal(data, expected)

    def test_select_not_in_index(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        selected = ds.select(number=[2, 3])
        expected = self.df.loc[[2]]
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_select_index_and_column(self):
        # See https://github.com/holoviz/holoviews/issues/6578
        frame = pd.DataFrame({"number": [1, 1, 2, 2], "color": ["red", "blue", "red", "blue"]})
        index = pd.MultiIndex.from_frame(frame, names=("number", "color"))
        df = pd.DataFrame({"cat": list("abab"), "values": range(4)}, index=index)
        ds = Dataset(df, kdims=["number"], vdims=["cat", "values"])
        selected = ds.select(number=[1], cat=["a"])
        expected = df[(df.index.get_level_values(0) == 1) & (df["cat"] == "a")]
        pd.testing.assert_frame_equal(selected.data, expected)

    def test_sample(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        sample = ds.interface.sample(ds, [1])
        assert sample.to_dict() == {'values': {(1, 'blue'): 1}}

        self.df.iloc[0, 0] = 1
        ds = Dataset(self.df, kdims=["number", "color"])
        sample = ds.interface.sample(ds, [1])
        assert sample.to_dict() == {'values': {(1, 'red'): 1, (1, 'blue'): 1}}

    def test_values(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        assert (ds.interface.values(ds, 'color') == ['red', 'blue', 'red', 'blue']).all()
        assert (ds.interface.values(ds, 'number') == [1, 1, 2, 2]).all()
        assert (ds.interface.values(ds, 'values') == [0, 1, 2, 3]).all()

    def test_reindex(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        df = ds.interface.reindex(ds, ['number', 'color'])
        assert df.index.names == ['number', 'color']

        df = ds.interface.reindex(ds, ['number'])
        assert df.index.names == ['number']

        df = ds.interface.reindex(ds, ['values'])
        assert df.index.names == ['values']

    def test_groupby_one_index(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        grouped = ds.groupby("number")
        assert list(grouped.keys()) == [1, 2]
        for k, v in grouped.items():
            pd.testing.assert_frame_equal(v.data, ds.select(number=k).data)

    def test_groupby_two_indexes(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        grouped = ds.groupby(["number", "color"])
        assert list(grouped.keys()) == list(self.df.index)
        for k, v in grouped.items():
            pd.testing.assert_frame_equal(v.data, ds.select(number=[k[0]], color=[k[1]]).data)

    def test_groupby_one_index_one_column(self):
        ds = Dataset(self.df, kdims=["number", "color"])
        grouped = ds.groupby('values')
        assert list(grouped.keys()) == [0, 1, 2, 3]
        for k, v in grouped.items():
            pd.testing.assert_frame_equal(v.data, ds.select(values=k).data)

    def test_regression_no_auto_index(self):
        # https://github.com/holoviz/holoviews/issues/6298

        plot = Scatter(self.df, kdims="number")
        np.testing.assert_equal(plot.dimension_values('number'), self.df.index.get_level_values('number'))


def test_no_subclasse_interface_applies():
    spd = pytest.importorskip("spatialpandas")
    square = spd.geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    sdf = spd.GeoDataFrame({"geometry": spd.GeoSeries([square, square]), "name": ["A", "B"]})
    assert PandasInterface.applies(sdf) is False
