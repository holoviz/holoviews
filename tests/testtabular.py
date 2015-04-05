"""
Unit tests of tabular elements
"""

from collections import OrderedDict
from holoviews import Table, ItemTable
from holoviews.element.comparison import ComparisonTestCase

class TestTable(ComparisonTestCase):


    def setUp(self):
        self.keys1 =   [('M',10), ('M',16), ('F',12)]
        self.values1 = [(15, 0.8), (18, 0.6), (10, 0.8)]
        self.key_dims1 = ['Gender', 'Age']
        self.val_dims1 = ['Weight', 'Height']

    def test_table_init(self):
        self.table1 =Table(zip(self.keys1, self.values1),
                           key_dimensions = self.key_dims1,
                           value_dimensions = self.val_dims1)

    def test_table_index_row_gender(self):
        table =Table(zip(self.keys1, self.values1),
                      key_dimensions = self.key_dims1,
                      value_dimensions = self.val_dims1)
        row = table['F',:]
        self.assertEquals(type(row), Table)
        self.assertEquals(row.data, OrderedDict([(('F', 12), (10, 0.8))]))

    def test_table_index_rows_gender(self):
        table =Table(zip(self.keys1, self.values1),
                      key_dimensions = self.key_dims1,
                      value_dimensions = self.val_dims1)
        row = table['M',:]
        self.assertEquals(type(row), Table)
        self.assertEquals(row.data,
                          OrderedDict([(('M', 10), (15, 0.8)), (('M', 16), (18, 0.6))]))

    def test_table_index_row_age(self):
        table =Table(zip(self.keys1, self.values1),
                      key_dimensions = self.key_dims1,
                      value_dimensions = self.val_dims1)
        row = table[:, 12]
        self.assertEquals(type(row), Table)
        self.assertEquals(row.data, OrderedDict([(('F', 12), (10, 0.8))]))

    def test_table_index_item_table(self):
        table =Table(zip(self.keys1, self.values1),
                      key_dimensions = self.key_dims1,
                      value_dimensions = self.val_dims1)
        itemtable = table['F', 12]
        self.assertEquals(type(itemtable), ItemTable)
        self.assertEquals(itemtable.data, OrderedDict([('Weight', 10), ('Height', 0.8)]))


    def test_table_index_value1(self):
        table =Table(zip(self.keys1, self.values1),
                      key_dimensions = self.key_dims1,
                      value_dimensions = self.val_dims1)
        self.assertEquals(table['F', 12, 'Weight'], 10)

    def test_table_index_value2(self):
        table =Table(zip(self.keys1, self.values1),
                      key_dimensions = self.key_dims1,
                      value_dimensions = self.val_dims1)
        self.assertEquals(table['F', 12, 'Height'], 0.8)

