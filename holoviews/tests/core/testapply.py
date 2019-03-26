import param

from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Params, ParamMethod


class ParamClass(param.Parameterized):

    label = param.String(default='Test')

    @param.depends('label')
    def apply_label(self, obj):
        return obj.relabel(self.label)

    @param.depends('label')
    def dynamic_label(self):
        return self.label + '!'


class TestApplyElement(ComparisonTestCase):

    def setUp(self):
        self.element = Curve([1, 2, 3])

    def test_element_apply_simple(self):
        applied = self.element.apply(lambda x: x.relabel('Test'))
        self.assertEqual(applied, self.element.relabel('Test'))

    def test_element_apply_method_as_string(self):
        applied = self.element.apply('relabel', label='Test')
        self.assertEqual(applied, self.element.relabel('Test'))

    def test_element_apply_with_kwarg(self):
        applied = self.element.apply(lambda x, label: x.relabel(label), label='Test')
        self.assertEqual(applied, self.element.relabel('Test'))

    def test_element_apply_not_dynamic_with_instance_param(self):
        pinst = ParamClass()
        applied = self.element.apply(lambda x, label: x.relabel(label), label=pinst.param.label, dynamic=False)
        self.assertEqual(applied, self.element.relabel('Test'))

    def test_element_apply_not_dynamic_with_method_string(self):
        pinst = ParamClass()
        applied = self.element.apply('relabel', dynamic=False, label=pinst.param.label)
        self.assertEqual(applied, self.element.relabel('Test'))

    def test_element_apply_not_dynamic_with_param_method(self):
        pinst = ParamClass()
        applied = self.element.apply(lambda x, label: x.relabel(label), label=pinst.dynamic_label, dynamic=False)
        self.assertEqual(applied, self.element.relabel('Test!'))

    def test_element_apply_dynamic(self):
        applied = self.element.apply(lambda x: x.relabel('Test'), dynamic=True)
        self.assertEqual(len(applied.streams), 0)
        self.assertEqual(applied[()], self.element.relabel('Test'))

    def test_element_apply_dynamic_with_kwarg(self):
        applied = self.element.apply(lambda x, label: x.relabel(label), dynamic=True, label='Test')
        self.assertEqual(len(applied.streams), 0)
        self.assertEqual(applied[()], self.element.relabel('Test'))

    def test_element_apply_dynamic_element_method(self):
        pinst = ParamClass()
        applied = self.element.apply(self.element.relabel, label=pinst.param.label)

        # Check stream
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, Params)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, ['label'])

        # Check results
        self.assertEqual(applied[()], self.element.relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[()], self.element.relabel('Another label'))

    def test_element_apply_dynamic_with_instance_param(self):
        pinst = ParamClass()
        applied = self.element.apply(lambda x, label: x.relabel(label), label=pinst.param.label)

        # Check stream
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, Params)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, ['label'])

        # Check results
        self.assertEqual(applied[()], self.element.relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[()], self.element.relabel('Another label'))

    def test_element_apply_param_method_with_dependencies(self):
        pinst = ParamClass()
        applied = self.element.apply(pinst.apply_label)

        # Check stream
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, ['label'])

        # Check results
        self.assertEqual(applied[()], self.element.relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[()], self.element.relabel('Another label'))

    def test_element_apply_dynamic_with_param_method(self):
        pinst = ParamClass()
        applied = self.element.apply(lambda x, label: x.relabel(label), label=pinst.dynamic_label)

        # Check stream
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, ['label'])

        # Check result
        self.assertEqual(applied[()], self.element.relabel('Test!'))
        pinst.label = 'Another label'
        self.assertEqual(applied[()], self.element.relabel('Another label!'))




class TestApplyDynamicMap(ComparisonTestCase):

    def setUp(self):
        self.element = Curve([1, 2, 3])
        self.dmap_unsampled = DynamicMap(lambda i: Curve([0, 1, i]), kdims='Y')
        self.dmap = self.dmap_unsampled.redim.values(Y=[0, 1, 2])

    def test_dmap_apply_not_dynamic_unsampled(self):
        with self.assertRaises(ValueError):
            self.dmap_unsampled.apply(lambda x: x.relabel('Test'), dynamic=False)
        
    def test_dmap_apply_not_dynamic(self):
        applied = self.dmap.apply(lambda x: x.relabel('Test'), dynamic=False)
        self.assertEqual(applied, HoloMap(self.dmap[[0, 1, 2]]).relabel('Test'))

    def test_dmap_apply_not_dynamic_with_kwarg(self):
        applied = self.dmap.apply(lambda x, label: x.relabel(label), dynamic=False, label='Test')
        self.assertEqual(applied, HoloMap(self.dmap[[0, 1, 2]]).relabel('Test'))

    def test_dmap_apply_not_dynamic_with_instance_param(self):
        pinst = ParamClass()
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label=pinst.param.label, dynamic=False)
        self.assertEqual(applied, HoloMap(self.dmap[[0, 1, 2]]).relabel('Test'))

    def test_dmap_apply_not_dynamic_with_param_method(self):
        pinst = ParamClass()
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label=pinst.dynamic_label, dynamic=False)
        self.assertEqual(applied, HoloMap(self.dmap[[0, 1, 2]]).relabel('Test!'))

    def test_dmap_apply_dynamic(self):
        applied = self.dmap.apply(lambda x: x.relabel('Test'))
        self.assertEqual(len(applied.streams), 0)
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))

    def test_element_apply_method_as_string(self):
        applied = self.dmap.apply('relabel', label='Test')
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))

    def test_dmap_apply_dynamic_with_kwarg(self):
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label='Test')
        self.assertEqual(len(applied.streams), 0)
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))

    def test_dmap_apply_dynamic_with_instance_param(self):
        pinst = ParamClass()
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label=pinst.param.label)

        # Check stream
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, Params)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, ['label'])

        # Check results
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[1], self.dmap[1].relabel('Another label'))

    def test_dmap_apply_method_as_string_with_instance_param(self):
        pinst = ParamClass()
        applied = self.dmap.apply('relabel', label=pinst.param.label)
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[1], self.dmap[1].relabel('Another label'))

    def test_dmap_apply_param_method_with_dependencies(self):
        pinst = ParamClass()
        applied = self.dmap.apply(pinst.apply_label)

        # Check stream
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, ['label'])

        # Check results
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[1], self.dmap[1].relabel('Another label'))

    def test_dmap_apply_dynamic_with_param_method(self):
        pinst = ParamClass()
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label=pinst.dynamic_label)

        # Check stream
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, ['label'])

        # Check result
        self.assertEqual(applied[1], self.dmap[1].relabel('Test!'))
        pinst.label = 'Another label'
        self.assertEqual(applied[1], self.dmap[1].relabel('Another label!'))
