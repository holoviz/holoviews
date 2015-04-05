import pickle
from collections import OrderedDict

import numpy as np

from holoviews import Image, Layout
from holoviews.element.comparison import ComparisonTestCase
from holoviews.interface.collector import ViewRef


class LayoutTest(ComparisonTestCase):

    def setUp(self):
        self.fixed_error = ("No attribute 'Test' in this AttrTree,"
                            " and none can be added because fixed=True")
        super(LayoutTest, self).setUp()

    def test_layout_init(self):
        Layout()

    def test_layout_getter(self):
        tr = Layout()
        self.assertEqual(isinstance(tr.Test.Path, Layout), True)

    def test_layout_getter_fixed(self):
        tr = Layout()
        tr.fixed = True
        try:
            tr.Test.Path
            raise AssertionError
        except AttributeError as e:
            self.assertEqual(str(e), self.fixed_error)

    def test_layout_setter(self):
        tr = Layout()
        tr.Test.Path = 42
        self.assertEqual(tr.Test.Path, 42)

    def test_layout_setter_fixed(self):
        tr = Layout()
        tr.fixed = True
        try:
            tr.Test.Path = 42
            raise AssertionError
        except AttributeError as e:
            self.assertEqual(str(e), self.fixed_error)

    def test_layout_shallow_fixed_setter(self):
        tr = Layout()
        tr.fixed = True
        try:
            tr.Test = 42
            raise AssertionError
        except AttributeError as e:
            self.assertEqual(str(e), self.fixed_error)

    def test_layout_toggle_fixed(self):
        tr = Layout()
        tr.fixed = True
        try:
            tr.Test = 42
            raise AssertionError
        except AttributeError as e:
            self.assertEqual(str(e), self.fixed_error)
        tr.fixed = False
        tr.Test = 42

    def test_layout_set_path(self):
        tr = Layout()
        tr.set_path(('Test', 'Path'), -42)
        self.assertEqual(tr.Test.Path, -42)


    def test_layout_update(self):
        tr1 = Layout()
        tr2 = Layout()
        tr1.Test1.Path1 = 42
        tr2.Test2.Path2 = -42
        tr1.update(tr2)
        self.assertEqual(tr1.Test1.Path1, 42)
        self.assertEqual(tr1.Test2.Path2, -42)


    def test_contains_child(self):
        tr = Layout()
        tr.Test.Path = 42
        self.assertEqual('Path' in tr.Test, True)

    def test_contains_tuple(self):
        tr = Layout()
        tr.Test.Path = 42
        self.assertEqual(('Test', 'Path') in tr, True)

    def test_simple_pickle(self):
        tr = Layout()
        dumped = pickle.dumps(tr)
        tr2 = pickle.loads(dumped)
        self.assertEqual(tr.data, OrderedDict())
        self.assertEqual(tr.data, tr2.data)

    def test_pickle_with_data(self):
        tr = Layout()
        tr.Example1.Data = 42
        tr.Example2.Data = 'some data'
        dumped = pickle.dumps(tr)
        tr2 = pickle.loads(dumped)
        self.assertEqual(tr.data, OrderedDict([(('Example1', 'Data'), 42),
                                                     (('Example2', 'Data'), 'some data')]))
        self.assertEqual(tr.data, tr2.data)



class ViewRefTest(ComparisonTestCase):

    def setUp(self):
        super(ViewRefTest, self).setUp()
        tree = Layout()
        tree.Example.Path1 = Image(np.random.rand(5,5))
        tree.Example.Path2 = Image(np.random.rand(5,5))
        self.tree = tree

    def test_resolve_constructor(self):
        ref = ViewRef('Example.Path1 * Example.Path2')
        overlay = ref.resolve(self.tree)
        self.assertEqual(len(overlay), 2)

    def test_resolve_setattr(self):
        ref = ViewRef().Example.Path1 * ViewRef().Example.Path2
        overlay = ref.resolve(self.tree)
        self.assertEqual(len(overlay), 2)

    def test_viewref_pickle(self):
        ref = ViewRef('Example.Path1 * Example.Path2')
        dumped = pickle.dumps(ref)
        ref2 = pickle.loads(dumped)
        self.assertEqual(ref.specification, [('Example', 'Path1'), ('Example', 'Path2')])
        self.assertEqual(ref.specification, ref2.specification)

