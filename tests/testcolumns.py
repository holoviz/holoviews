"""
Tests for the Columns Element types.
"""

import pandas as pd

import numpy as np
from holoviews import OrderedDict, Columns, Curve, ItemTable, NdElement, HoloMap
from holoviews.element.comparison import ComparisonTestCase


class ColumnsNdElementTest(ComparisonTestCase):
    """
    Test for the Chart baseclass methods.
    """

    def setUp(self):
        self.datatype = Columns.datatype
        Columns.datatype = ['dictionary', 'array']
        self.xs = range(11)
        self.ys = np.linspace(0, 1, 11)
        self.zs = np.sin(self.xs)
        self.keys1 =   [('M',10), ('M',16), ('F',12)]
        self.values1 = [(15, 0.8), (18, 0.6), (10, 0.8)]
        self.kdims = ['Gender', 'Age']
        self.vdims = ['Weight', 'Height']
        self.columns = Columns(dict(zip(self.xs, self.ys)),
                               kdims=['x'], vdims=['y'])

    def tearDown(self):
        Columns.datatype = self.datatype

    def test_columns_sort_vdim(self):
        columns = Columns(OrderedDict(zip(self.xs, -self.ys)),
                          kdims=['x'], vdims=['y'])
        columns_sorted = Columns(OrderedDict(zip(self.xs[::-1], -self.ys[::-1])),
                                 kdims=['x'], vdims=['y'])
        self.assertEqual(columns.sort('y'), columns_sorted)

    def test_columns_sort_heterogeneous_string(self):
        columns = Columns(zip(self.keys1, self.values1),
                        kdims=self.kdims, vdims=self.vdims)
        keys =   [('F',12), ('M',10), ('M',16)]
        values = [(10, 0.8), (15, 0.8), (18, 0.6)]
        columns_sorted = Columns(zip(keys, values),
                                 kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(columns.sort(), columns_sorted)

    def test_columns_shape(self):
        self.assertEqual(self.columns.shape, (11, 2))

    def test_columns_range(self):
        self.assertEqual(self.columns.range('y'), (0., 1.))

    def test_columns_odict_construct(self):
        columns = Columns(OrderedDict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(columns.data, NdElement))

    def test_columns_closest(self):
        closest = self.columns.closest([0.51, 1, 9.9])
        self.assertEqual(closest, [1., 1., 10.])

    def test_columns_dict_construct(self):
        self.assertTrue(isinstance(self.columns.data, NdElement))

    def test_columns_ndelement_construct(self):
        columns = Columns(NdElement(zip(self.xs, self.ys)))
        self.assertTrue(isinstance(columns.data, NdElement))

    def test_columns_items_construct(self):
        columns = Columns(zip(self.keys1, self.values1),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertTrue(isinstance(columns.data, NdElement))

    def test_columns_sample(self):
        samples = self.columns.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 0.5, 1]))

    def test_columns_index_row_gender(self):
        table = Columns(zip(self.keys1, self.values1),
                        kdims=self.kdims, vdims=self.vdims)
        indexed = Columns(OrderedDict([(('F', 12), (10, 0.8))]),
                          kdims=self.kdims, vdims=self.vdims)
        row = table['F',:]
        self.assertEquals(row, indexed)

    def test_columns_index_rows_gender(self):
        table = Columns(zip(self.keys1, self.values1),
                        kdims=self.kdims, vdims=self.vdims)
        row = table['M',:]
        indexed = Columns(OrderedDict([(('M', 10), (15, 0.8)),
                                       (('M', 16), (18, 0.6))]),
                             kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(row, indexed)

    def test_columns_index_row_age(self):
        table = Columns(zip(self.keys1, self.values1),
                        kdims=self.kdims, vdims=self.vdims)
        indexed = Columns(OrderedDict([(('F', 12), (10, 0.8))]),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(table[:, 12], indexed)

    def test_columns_index_item_table(self):
        table = Columns(zip(self.keys1, self.values1),
                        kdims=self.kdims, vdims=self.vdims)
        indexed = Columns(OrderedDict([(('F', 12), (10, 0.8))]),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(table['F', 12], indexed)


    def test_columns_index_value1(self):
        table = Columns(zip(self.keys1, self.values1),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(table['F', 12, 'Weight'], 10)

    def test_columns_index_value2(self):
        table = Columns(zip(self.keys1, self.values1),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(table['F', 12, 'Height'], 0.8)

    def test_columns_getitem_column(self):
        self.compare_arrays(self.columns['y'], self.ys)

    def test_columns_add_dimensions_value(self):
        table = self.columns.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(len(table)))

    def test_columns_add_dimensions_values(self):
        table = self.columns.add_dimension('z', 1, range(1,12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))

    def test_columns_collapse(self):
        collapsed = HoloMap({i: Columns(dict(zip(self.xs, self.ys*i)), kdims=['x'], vdims=['y'])
                             for i in range(10)}, kdims=['z']).collapse('z', np.mean)
        self.compare_columns(collapsed, Columns(zip(zip(self.xs), self.ys*4.5),
                                                kdims=['x'], vdims=['y']))

    def test_columns_1d_reduce(self):
        self.assertEqual(self.columns.reduce('x', np.mean), np.float64(0.5))

    def test_columns_2d_reduce(self):
        columns = Columns(zip(zip(self.xs, self.ys), self.zs),
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(np.array(columns.reduce(['x', 'y'], np.mean)),
                         np.array(0.12828985192891))

    def test_columns_2d_partial_reduce(self):
        columns = Columns(zip(zip(self.xs, self.ys), self.zs),
                          kdims=['x', 'y'], vdims=['z'])
        reduced = Columns(zip(zip(self.xs), self.zs),
                          kdims=['x'], vdims=['z'])
        self.assertEqual(columns.reduce(['y'], np.mean), reduced)

    def test_columns_heterogeneous_reduce(self):
        columns = Columns(zip(self.keys1, self.values1), kdims=self.kdims,
                          vdims=self.vdims)
        reduced = Columns(zip([k[1:] for k in self.keys1], self.values1),
                          kdims=self.kdims[1:], vdims=self.vdims)
        self.assertEqual(columns.reduce(['Gender'], np.mean), reduced)

    def test_columns_heterogeneous_reduce2d(self):
        columns = Columns(zip(self.keys1, self.values1), kdims=self.kdims,
                          vdims=self.vdims)
        reduced = Columns([((), (14.333333333333334, 0.73333333333333339))], kdims=[], vdims=self.vdims)
        self.assertEqual(columns.reduce(function=np.mean), reduced)

    def test_column_heterogeneous_aggregate(self):
        columns = Columns(zip(self.keys1, self.values1), kdims=self.kdims,
                          vdims=self.vdims)
        aggregated = Columns(OrderedDict([('M', (16.5, 0.7)), ('F', (10., 0.8))]),
                             kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_columns(columns.aggregate(['Gender'], np.mean), aggregated)

    def test_columns_2d_aggregate_partial(self):
        columns = Columns(zip(zip(self.xs, self.ys), self.zs),
                          kdims=['x', 'y'], vdims=['z'])
        reduced = Columns(zip(zip(self.xs), self.zs),
                          kdims=['x'], vdims=['z'])
        self.assertEqual(columns.aggregate(['x'], np.mean), reduced)

    def test_columns_array(self):
        self.assertEqual(self.columns.array(), np.column_stack([self.xs, self.ys]))


class ColumnsNdArrayTest(ComparisonTestCase):

    def setUp(self):
        self.xs = range(11)
        self.ys = np.linspace(0, 1, 11)
        self.zs = np.sin(self.xs)
        self.columns = Columns((self.xs, self.ys), kdims=['x'], vdims=['y'])

    def test_columns_shape(self):
        self.assertEqual(self.columns.shape, (11, 2))

    def test_columns_range(self):
        self.assertEqual(self.columns.range('y'), (0., 1.))

    def test_columns_closest(self):
        closest = self.columns.closest([0.51, 1, 9.9])
        self.assertEqual(closest, [1., 1., 10.])

    def test_columns_values_construct(self):
        columns = Columns(self.ys)
        self.assertTrue(isinstance(columns.data, np.ndarray))

    def test_columns_tuple_construct(self):
        columns = Columns((self.xs, self.ys))
        self.assertTrue(isinstance(columns.data, np.ndarray))

    def test_columns_array_construct(self):
        columns = Columns(np.column_stack([self.xs, self.ys]))
        self.assertTrue(isinstance(columns.data, np.ndarray))

    def test_columns_tuple_list_construct(self):
        columns = Columns(zip(self.xs, self.ys))
        self.assertTrue(isinstance(columns.data, np.ndarray))

    def test_columns_sort_vdim(self):
        columns = Columns((self.xs, -self.ys), kdims=['x'], vdims=['y'])
        columns_sorted = Columns((self.xs[::-1], -self.ys[::-1]),
                                 kdims=['x'], vdims=['y'])
        self.assertEqual(columns.sort('y'), columns_sorted)

    def test_columns_index(self):
        self.assertEqual(self.columns[5], self.ys[5])

    def test_columns_slice(self):
        columns_slice = Columns(zip(range(5, 9), np.linspace(0.5,0.8, 4)),
                                kdims=['x'], vdims=['y'])
        self.assertEqual(self.columns[5:9], columns_slice)

    def test_columns_closest(self):
        closest = self.columns.closest([0.51, 1, 9.9])
        self.assertEqual(closest, [1., 1., 10.])

    def test_columns_getitem_column(self):
        self.compare_arrays(self.columns['y'], self.ys)

    def test_columns_sample(self):
        samples = self.columns.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 0.5, 1]))

    def test_columns_add_dimensions_value(self):
        table = Columns((self.xs, self.ys),
                        kdims=['x'], vdims=['y'])
        table = table.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(len(table)))

    def test_columns_add_dimensions_values(self):
        table = Columns((self.xs, self.ys),
                        kdims=['x'], vdims=['y'])
        table = table.add_dimension('z', 1, range(1,12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))

    def test_columns_collapse(self):
        collapsed = HoloMap({i: Columns((self.xs, self.ys*i), kdims=['x'], vdims=['y'])
                             for i in range(10)}, kdims=['z']).collapse('z', np.mean)
        self.compare_columns(collapsed, Columns((self.xs, self.ys*4.5), kdims=['x'], vdims=['y']))

    def test_columns_1d_reduce(self):
        columns = Columns((self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertEqual(columns.reduce('x', np.mean), np.float64(0.5))

    def test_columns_2d_reduce(self):
        columns = Columns((self.xs, self.ys, self.zs), kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(np.array(columns.reduce(['x', 'y'], np.mean)),
                         np.array(0.12828985192891))

    def test_columns_2d_partial_reduce(self):
        columns = Columns((self.xs, self.ys, self.zs), kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(columns.reduce(['y'], np.mean),
                         Columns((self.xs, self.zs), kdims=['x'], vdims=['z']))

    def test_columns_2d_aggregate_partial(self):
        columns = Columns((self.xs, self.ys, self.zs), kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(columns.aggregate(['x'], np.mean),
                         Columns((self.xs, self.zs), kdims=['x'], vdims=['z']))

    def test_columns_array(self):
        self.assertEqual(self.columns.array(), np.column_stack([self.xs, self.ys]))


class ColumnsDFrameTest(ComparisonTestCase):

    def setUp(self):
        self.datatype = Columns.datatype
        Columns.datatype = ['dataframe']
        self.column_data = [('M',10, 15, 0.8), ('M',16, 18, 0.6),
                            ('F',12, 10, 0.8)]
        self.kdims = ['Gender', 'Age']
        self.vdims = ['Weight', 'Height']
        self.xs = range(11)
        self.ys = np.linspace(0, 1, 11)
        self.zs = np.sin(self.xs)
        self.columns = Columns(pd.DataFrame({'x': self.xs, 'y': self.ys}),
                               kdims=['x'], vdims=['y'])

    def tearDown(self):
        Columns.datatype = self.datatype

    def test_columns_range(self):
        self.assertEqual(self.columns.range('y'), (0., 1.))

    def test_columns_shape(self):
        self.assertEqual(self.columns.shape, (11, 2))

    def test_columns_closest(self):
        closest = self.columns.closest([0.51, 1, 9.9])
        self.assertEqual(closest, [1., 1., 10.])

    def test_columns_sample(self):
        samples = self.columns.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 0.5, 1]))

    def test_columns_df_construct(self):
        self.assertTrue(isinstance(self.columns.data, pd.DataFrame))

    def test_columns_tuple_list_construct(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        self.assertTrue(isinstance(self.columns.data, pd.DataFrame))

    def test_columns_slice(self):
        data = [('x', range(5, 9)), ('y', np.linspace(0.5, 0.8, 4))]
        columns_slice = Columns(pd.DataFrame.from_items(data),
                                kdims=['x'], vdims=['y'])
        self.assertEqual(self.columns[5:9], columns_slice)

    def test_columns_index_row_gender(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        row = columns['F',:]
        self.assertEquals(type(row), Columns)
        self.compare_columns(row, Columns(self.column_data[2:],
                                          kdims=self.kdims,
                                          vdims=self.vdims))

    def test_columns_index_rows_gender(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        row = columns['M',:]
        self.assertEquals(type(row), Columns)
        self.compare_columns(row, Columns(self.column_data[:2],
                                          kdims=self.kdims,
                                          vdims=self.vdims))

    def test_columns_index_row_age(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        row = columns[:, 12]
        self.assertEquals(type(row), Columns)
        self.compare_columns(row, Columns(self.column_data[2:],
                                          kdims=self.kdims,
                                          vdims=self.vdims))

    def test_columns_index_single_row(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        row = columns['F', 12]
        self.assertEquals(type(row), Columns)
        self.compare_columns(row, Columns(self.column_data[2:],
                                          kdims=self.kdims,
                                          vdims=self.vdims))

    def test_columns_index_value1(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        self.assertEquals(columns['F', 12, 'Weight'], 10)

    def test_columns_index_value2(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        self.assertEquals(columns['F', 12, 'Height'], 0.8)

    def test_columns_sort_vdim(self):
        columns = Columns(pd.DataFrame({'x': self.xs, 'y': -self.ys}),
                          kdims=['x'], vdims=['y'])
        columns_sorted = Columns(pd.DataFrame({'x': self.xs[::-1], 'y': -self.ys[::-1]}),
                                 kdims=['x'], vdims=['y'])
        self.assertEqual(columns.sort('y'), columns_sorted)

    def test_columns_sort_heterogeneous_string(self):
        columns = Columns(self.column_data, kdims=self.kdims, vdims=self.vdims)
        columns_sorted = Columns([self.column_data[i] for i in [2, 0, 1]],
                                 kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(columns.sort(), columns_sorted)

    def test_columns_add_dimensions_value(self):
        columns = self.columns.add_dimension('z', 1, 0)
        self.assertEqual(columns.kdims[1], 'z')
        self.compare_arrays(columns.dimension_values('z'), np.zeros(len(columns)))

    def test_columns_add_dimensions_values(self):
        columns = self.columns.add_dimension('z', 1, range(1,12))
        self.assertEqual(columns.kdims[1], 'z')
        self.compare_arrays(columns.dimension_values('z'), np.array(list(range(1,12))))

    def test_columns_getitem_column(self):
        self.compare_arrays(self.columns['y'], self.ys)

    def test_columns_collapse(self):
        collapsed = HoloMap({i: Columns(pd.DataFrame({'x': self.xs, 'y': self.ys*i}), kdims=['x'], vdims=['y'])
                             for i in range(10)}, kdims=['z']).collapse('z', np.mean)
        self.compare_columns(collapsed, Columns(pd.DataFrame({'x': self.xs, 'y': self.ys*4.5}), kdims=['x'], vdims=['y']))

    def test_columns_1d_reduce(self):
        self.assertEqual(self.columns.reduce('x', np.mean), np.float64(0.5))

    def test_columns_2d_reduce(self):
        columns = Columns(pd.DataFrame({'x': self.xs, 'y': self.ys, 'z': self.zs}),
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(np.array(columns.reduce(['x', 'y'], np.mean)),
                         np.array(0.12828985192891))

    def test_columns_2d_partial_reduce(self):
        columns = Columns(pd.DataFrame({'x': self.xs, 'y': self.ys, 'z': self.zs}),
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(columns.reduce(['y'], np.mean),
                         Columns(pd.DataFrame({'x': self.xs, 'z': self.zs}),
                                 kdims=['x'], vdims=['z']))

    def test_columns_heterogeneous_reduce(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        reduced_data = pd.DataFrame([(10, 15, 0.8), (12, 10, 0.8), (16, 18, 0.6)],
                                    columns=columns.dimensions(label=True)[1:])
        reduced = Columns(reduced_data, kdims=self.kdims[1:],
                          vdims=self.vdims)
        self.assertEqual(columns.reduce(['Gender'], np.mean), reduced)

    def test_columns_heterogeneous_reduce2d(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        reduced_data = pd.DataFrame([d[1:] for d in self.column_data],
                                    columns=columns.dimensions(label=True)[1:])
        reduced = Columns(pd.DataFrame([(14.333333333333334, 0.73333333333333339)], columns=self.vdims),
                          kdims=[], vdims=self.vdims)
        self.assertEqual(columns.reduce(function=np.mean), reduced)


    def test_columns_groupby(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        cols = self.kdims + self.vdims
        group1 = pd.DataFrame(self.column_data[:2], columns=cols)
        group2 = pd.DataFrame(self.column_data[2:], columns=cols)
        grouped = HoloMap({'M': Columns(group1, kdims=['Age'], vdims=self.vdims),
                           'F': Columns(group2, kdims=['Age'], vdims=self.vdims)},
                          kdims=['Gender'])
        self.assertEqual(columns.groupby(['Gender']), grouped)

    def test_columns_heterogeneous_aggregate(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        aggregated = Columns(pd.DataFrame([('F', 10., 0.8), ('M', 16.5, 0.7)],
                                          columns=['Gender']+self.vdims),
                             kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_columns(columns.aggregate(['Gender'], np.mean), aggregated)

    def test_columns_2d_partial_reduce(self):
        columns = Columns(pd.DataFrame({'x': self.xs, 'y': self.ys, 'z': self.zs}),
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(columns.aggregate(['x'], np.mean),
                         Columns(pd.DataFrame({'x': self.xs, 'z': self.zs}),
                                 kdims=['x'], vdims=['z']))

    def test_columns_array(self):
        self.assertEqual(self.columns.array(), np.column_stack([self.xs, self.ys]))
