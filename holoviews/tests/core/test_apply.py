import numpy as np
import pandas as pd
import param

from panel.widgets import RadioButtonGroup, TextInput

from holoviews import Dataset, util
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import Curve, Image
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import ParamMethod, Params


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

    def test_element_apply_dynamic_with_widget_kwarg(self):
        text = TextInput()
        applied = self.element.apply(lambda x, label: x.relabel(label), label=text)
        self.assertEqual(len(applied.streams), 1)
        self.assertEqual(applied[()].label, '')
        text.value = 'Test'
        self.assertEqual(applied[()].label, 'Test')

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
        self.assertEqual(stream.parameters, [pinst.param.label])

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
        self.assertEqual(stream.parameters, [pinst.param.label])

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
        self.assertEqual(stream.parameters, [pinst.param.label])

        # Check results
        self.assertEqual(applied[()], self.element.relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[()], self.element.relabel('Another label'))

    def test_element_apply_function_with_dependencies(self):
        pinst = ParamClass()

        @param.depends(pinst.param.label)
        def get_label(label):
            return label + '!'

        applied = self.element.apply('relabel', label=get_label)

        # Check stream
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, Params)
        self.assertEqual(stream.parameters, [pinst.param.label])

        # Check results
        self.assertEqual(applied[()], self.element.relabel('Test!'))

        # Ensure subscriber gets called
        stream.add_subscriber(lambda **kwargs: applied[()])
        pinst.label = 'Another label'
        self.assertEqual(applied.last, self.element.relabel('Another label!'))

    def test_element_apply_function_with_dependencies_non_dynamic(self):
        pinst = ParamClass()

        @param.depends(pinst.param.label)
        def get_label(label):
            return label + '!'

        applied = self.element.apply('relabel', dynamic=False, label=get_label)
        self.assertEqual(applied, self.element.relabel('Test!'))

    def test_element_apply_dynamic_with_param_method(self):
        pinst = ParamClass()
        applied = self.element.apply(lambda x, label: x.relabel(label), label=pinst.dynamic_label)

        # Check stream
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, [pinst.param.label])

        # Check result
        self.assertEqual(applied[()], self.element.relabel('Test!'))
        pinst.label = 'Another label'
        self.assertEqual(applied[()], self.element.relabel('Another label!'))

    def test_holomap_apply_with_method(self):
        hmap = HoloMap({i: Image(np.array([[i, 2], [3, 4]])) for i in range(3)})
        reduced = hmap.apply.reduce(x=np.min)

        expected = HoloMap({i: Curve([(-0.25, 3), (0.25, i)], 'y', 'z') for i in range(3)})
        self.assertEqual(reduced, expected)



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
        self.assertEqual(stream.parameters, [pinst.param.label])

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
        self.assertEqual(stream.parameters, [pinst.param.label])

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
        self.assertEqual(stream.parameters, [pinst.param.label])

        # Check result
        self.assertEqual(applied[1], self.dmap[1].relabel('Test!'))
        pinst.label = 'Another label'
        self.assertEqual(applied[1], self.dmap[1].relabel('Another label!'))


def test_nested_widgets():
    df = pd._testing.makeDataFrame()
    column = RadioButtonGroup(value="A", options=list("ABC"))
    ds = Dataset(df)
    transform = util.transform.df_dim("*").groupby(["D", column]).mean()

    params = list(transform.params.values())
    assert len(params) == 1
    assert params[0] == column.param.value

    df1 = transform.apply(ds, keep_index=True, compute=False)
    df2 = df.groupby(["D", "A"]).mean()
    pd.testing.assert_frame_equal(df1, df2)
