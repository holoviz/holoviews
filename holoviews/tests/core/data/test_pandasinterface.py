from unittest import SkipTest

import numpy as np

try:
    import pandas as pd
except:
    raise SkipTest("Could not import pandas, skipping PandasInterface tests.")

from holoviews.core.dimension import Dimension
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.spaces import HoloMap
from holoviews.element import Scatter, Points, Distribution


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


class PandasInterfaceTests(BasePandasInterfaceTests):

    datatype = 'dataframe'
    data_type = pd.DataFrame

    __test__ = True
