import param

from holoviews.core.operation import Operation
from holoviews.element import Curve
from holoviews.streams import Params, Stream
from holoviews.testing import assert_element_equal


class ExampleOperation(Operation):

    label = param.String()

    def _process(self, obj, key=None):
        return obj.relabel(self.p.label)


class ParamClass(param.Parameterized):

    label = param.String(default='Test')

    @param.depends('label')
    def dynamic_label(self):
        return self.label + '!'


class TestOperationBroadcast:

    def test_element_dynamic_with_streams(self):
        curve = Curve([1, 2, 3])
        applied = Operation(curve, dynamic=True, streams=[Stream])
        assert len(applied.streams) == 1
        assert isinstance(applied.streams[0], Stream)
        assert_element_equal(applied[()], curve)

    def test_element_not_dynamic_despite_streams(self):
        curve = Curve([1, 2, 3])
        applied = Operation(curve, dynamic=False, streams=[Stream])
        assert_element_equal(applied, curve)

    def test_element_dynamic_with_instance_param(self):
        curve = Curve([1, 2, 3])
        inst = ParamClass(label='Test')
        applied = ExampleOperation(curve, label=inst.param.label)
        assert len(applied.streams) == 1
        assert isinstance(applied.streams[0], Params)
        assert applied.streams[0].parameters == [inst.param.label]
        assert_element_equal(applied[()], curve.relabel('Test'))

    def test_element_dynamic_with_param_method(self):
        curve = Curve([1, 2, 3])
        inst = ParamClass(label='Test')
        applied = ExampleOperation(curve, label=inst.dynamic_label)
        assert len(applied.streams) == 1
        assert isinstance(applied.streams[0], Params)
        assert applied.streams[0].parameters == [inst.param.label]
        assert_element_equal(applied[()], curve.relabel('Test!'))
        inst.label = 'New label'
        assert_element_equal(applied[()], curve.relabel('New label!'))

    def test_element_not_dynamic_with_instance_param(self):
        curve = Curve([1, 2, 3])
        inst = ParamClass(label='Test')
        applied = ExampleOperation(curve, dynamic=False, label=inst.param.label)
        assert_element_equal(applied, curve.relabel('Test'))

    def test_element_not_dynamic_with_param_method(self):
        curve = Curve([1, 2, 3])
        inst = ParamClass(label='Test')
        applied = ExampleOperation(curve, dynamic=False, label=inst.dynamic_label)
        assert_element_equal(applied, curve.relabel('Test!'))
