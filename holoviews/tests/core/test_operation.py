import param

from holoviews.core.operation import Operation
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Stream, Params


class TestOperation(Operation):

    label = param.String()

    def _process(self, obj, key=None):
        return obj.relabel(self.p.label)


class ParamClass(param.Parameterized):

    label = param.String(default='Test')

    @param.depends('label')
    def dynamic_label(self):
        return self.label + '!'


class TestOperationBroadcast(ComparisonTestCase):

    def test_element_dynamic_with_streams(self):
        curve = Curve([1, 2, 3])
        applied = Operation(curve, dynamic=True, streams=[Stream])
        self.assertEqual(len(applied.streams), 1)
        self.assertIsInstance(applied.streams[0], Stream)
        self.assertEqual(applied[()], curve)

    def test_element_not_dynamic_despite_streams(self):
        curve = Curve([1, 2, 3])
        applied = Operation(curve, dynamic=False, streams=[Stream])
        self.assertEqual(applied, curve)

    def test_element_dynamic_with_instance_param(self):
        curve = Curve([1, 2, 3])
        inst = ParamClass(label='Test')
        applied = TestOperation(curve, label=inst.param.label)
        self.assertEqual(len(applied.streams), 1)
        self.assertIsInstance(applied.streams[0], Params)
        self.assertEqual(applied.streams[0].parameters, [inst.param.label])
        self.assertEqual(applied[()], curve.relabel('Test'))

    def test_element_dynamic_with_param_method(self):
        curve = Curve([1, 2, 3])
        inst = ParamClass(label='Test')
        applied = TestOperation(curve, label=inst.dynamic_label)
        self.assertEqual(len(applied.streams), 1)
        self.assertIsInstance(applied.streams[0], Params)
        self.assertEqual(applied.streams[0].parameters, [inst.param.label])
        self.assertEqual(applied[()], curve.relabel('Test!'))
        inst.label = 'New label'
        self.assertEqual(applied[()], curve.relabel('New label!'))

    def test_element_not_dynamic_with_instance_param(self):
        curve = Curve([1, 2, 3])
        inst = ParamClass(label='Test')
        applied = TestOperation(curve, dynamic=False, label=inst.param.label)
        self.assertEqual(applied, curve.relabel('Test'))

    def test_element_not_dynamic_with_param_method(self):
        curve = Curve([1, 2, 3])
        inst = ParamClass(label='Test')
        applied = TestOperation(curve, dynamic=False, label=inst.dynamic_label)
        self.assertEqual(applied, curve.relabel('Test!'))
