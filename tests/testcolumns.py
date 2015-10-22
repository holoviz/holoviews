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
        self.xs = range(11)
        self.ys = np.linspace(0, 1, 11)
        self.keys1 =   [('M',10), ('M',16), ('F',12)]
        self.values1 = [(15, 0.8), (18, 0.6), (10, 0.8)]
        self.key_dims1 = ['Gender', 'Age']
        self.val_dims1 = ['Weight', 'Height']
        self.columns = Columns(dict(zip(self.xs, self.ys)),
                               kdims=['x'], vdims=['y'])

    def test_columns_odict_construct(self):
        columns = Columns(OrderedDict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(columns.data, NdElement))

    def test_columns_dict_construct(self):
        self.assertTrue(isinstance(self.columns.data, NdElement))

    def test_columns_ndelement_construct(self):
        columns = Columns(NdElement(zip(self.xs, self.ys)))
        self.assertTrue(isinstance(columns.data, NdElement))

    def test_columns_items_construct(self):
        columns = Columns(zip(self.keys1, self.values1),
                          kdims = self.key_dims1,
                          vdims = self.val_dims1)
        self.assertTrue(isinstance(columns.data, NdElement))

    def test_columns_index_row_gender(self):
        table =Columns(zip(self.keys1, self.values1),
                      kdims = self.key_dims1,
                      vdims = self.val_dims1)
        row = table['F',:]
        self.assertEquals(type(row), Columns)
        self.assertEquals(row.data.data, OrderedDict([(('F', 12), (10, 0.8))]))

    def test_columns_index_rows_gender(self):
        table =Columns(zip(self.keys1, self.values1),
                      kdims = self.key_dims1,
                      vdims = self.val_dims1)
        row = table['M',:]
        self.assertEquals(type(row), Columns)
        self.assertEquals(row.data.data,
                          OrderedDict([(('M', 10), (15, 0.8)), (('M', 16), (18, 0.6))]))

    def test_columns_index_row_age(self):
        table =Columns(zip(self.keys1, self.values1),
                      kdims = self.key_dims1,
                      vdims = self.val_dims1)
        row = table[:, 12]
        self.assertEquals(type(row), Columns)
        self.assertEquals(row.data.data, OrderedDict([(('F', 12), (10, 0.8))]))

    def test_columns_index_item_table(self):
        table =Columns(zip(self.keys1, self.values1),
                      kdims = self.key_dims1,
                      vdims = self.val_dims1)
        itemtable = table['F', 12]
        self.assertEquals(type(itemtable), Columns)
        self.assertEquals(itemtable.data.data, OrderedDict([(('F', 12), (10, 0.8))]))


    def test_columns_index_value1(self):
        table =Columns(zip(self.keys1, self.values1),
                      kdims = self.key_dims1,
                      vdims = self.val_dims1)
        self.assertEquals(table['F', 12, 'Weight'], 10)

    def test_columns_index_value2(self):
        table =Columns(zip(self.keys1, self.values1),
                      kdims = self.key_dims1,
                      vdims = self.val_dims1)
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
        self.compare_columns(collapsed, Columns(dict(zip(self.xs, self.ys*4.5)),
                                                kdims=['x'], vdims=['y']))


class ColumnsNdArrayTest(ComparisonTestCase):

    def setUp(self):
        self.xs = range(11)
        self.ys = np.linspace(0, 1, 11)
        self.columns = Columns((self.xs, self.ys), kdims=['x'], vdims=['y'])

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



class ColumnsDFrameTest(ComparisonTestCase):

    def setUp(self):
        self.column_data = [('M',10, 15, 0.8), ('M',16, 18, 0.6),
                            ('F',12, 10, 0.8)]
        self.kdims = ['Gender', 'Age']
        self.vdims = ['Weight', 'Height']
        self.xs = range(11)
        self.ys = np.linspace(0, 1, 11)
        self.columns = Columns(pd.DataFrame({'x': self.xs, 'y': self.ys}),
                               kdims=['x'], vdims=['y'])

    def test_columns_df_construct(self):
        self.assertTrue(isinstance(self.columns.data, pd.DataFrame))

    def test_columns_tuple_list_construct(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        self.assertTrue(isinstance(self.columns.data, pd.DataFrame))

    def test_columns_index_row_gender(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        row = columns['F',:]
        self.assertEquals(type(row), Columns)
        self.compare_columns(row.clone(row.data.reset_index(drop=True)),
                             Columns(self.column_data[2:],
                                     kdims=self.kdims,
                                     vdims=self.vdims))

    def test_columns_index_rows_gender(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        row = columns['M',:]
        self.assertEquals(type(row), Columns)
        self.compare_columns(row.clone(row.data.reset_index(drop=True)),
                             Columns(self.column_data[:2],
                                     kdims=self.kdims,
                                     vdims=self.vdims))

    def test_columns_index_row_age(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        row = columns[:, 12]
        self.assertEquals(type(row), Columns)
        self.compare_columns(row.clone(row.data.reset_index(drop=True)),
                             Columns(self.column_data[2:],
                                     kdims=self.kdims,
                                     vdims=self.vdims))

    def test_columns_index_single_row(self):
        columns = Columns(self.column_data, kdims=self.kdims,
                          vdims=self.vdims)
        row = columns['F', 12]
        self.assertEquals(type(row), Columns)
        self.compare_columns(row.clone(row.data.reset_index(drop=True)),
                             Columns(self.column_data[2:],
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

    def test_columns_add_dimensions_value(self):
        table = self.columns.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(len(table)))

    def test_columns_add_dimensions_values(self):
        table = self.columns.add_dimension('z', 1, range(1,12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))

    def test_columns_getitem_column(self):
        self.compare_arrays(self.columns['y'], self.ys)

    def test_columns_collapse(self):
        collapsed = HoloMap({i: Columns(pd.DataFrame({'x': self.xs, 'y': self.ys*i}), kdims=['x'], vdims=['y'])
                             for i in range(10)}, kdims=['z']).collapse('z', np.mean)
        self.compare_columns(collapsed, Columns(pd.DataFrame({'x': self.xs, 'y': self.ys*4.5}), kdims=['x'], vdims=['y']))
