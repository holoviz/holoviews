"""
Unit test of the streams system
"""
from collections import defaultdict
from unittest import SkipTest

import pandas as pd
import param
import pytest
from panel.widgets import IntSlider

import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import NUMPY_GE_2_0_0, PARAM_VERSION
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim

from .utils import LoggingComparisonTestCase

PARAM_GE_2_0_0 = PARAM_VERSION >= (2, 0, 0)

def test_all_stream_parameters_constant():
    all_stream_cls = [v for v in globals().values() if
                      isinstance(v, type) and issubclass(v, Stream)]
    for stream_cls in all_stream_cls:
        for name, p in stream_cls.param.objects().items():
            if name == 'name': continue
            if p.constant != True:
                raise TypeError(f'Parameter {name} of stream {stream_cls.__name__} not declared constant')


def test_all_linked_stream_parameters_owners():
    "Test to ensure operations can accept parameters in streams dictionary"
    stream_classes = param.concrete_descendents(LinkedStream)
    for stream_class in stream_classes.values():
        for name, p in stream_class.param.objects().items():
            if name != 'name' and (p.owner != stream_class):
                msg = ("Linked stream %r has parameter %r which is "
                       "inherited from %s. Parameter needs to be redeclared "
                       "in the class definition of this linked stream.")
                raise Exception(msg % (stream_class, name, p.owner))

class TestStreamsDefine(ComparisonTestCase):

    def setUp(self):
        self.XY = Stream.define('XY', x=0.0, y=5.0)
        self.TypesTest = Stream.define('TypesTest',
                                       t=True,
                                       u=0,
                                       v=1.2,
                                       w= (1,'a'),
                                       x='string',
                                       y= [],
                                       z = np.array([1,2,3]))

        test_param = param.Integer(default=42, doc='Test docstring')
        self.ExplicitTest = Stream.define('ExplicitTest',
                                          test=test_param)

    def test_XY_types(self):
        self.assertEqual(isinstance(self.XY.param['x'], param.Number),True)
        self.assertEqual(isinstance(self.XY.param['y'], param.Number),True)

    def test_XY_defaults(self):
        self.assertEqual(self.XY.param['x'].default,0.0)
        self.assertEqual(self.XY.param['y'].default, 5.0)

    def test_XY_instance(self):
        xy = self.XY(x=1,y=2)
        self.assertEqual(xy.x, 1)
        self.assertEqual(xy.y, 2)

    def test_XY_set_invalid_class_x(self):
        if PARAM_GE_2_0_0:
            regexp = "Number parameter 'XY.x' only takes numeric values"
        else:
            regexp = "Parameter 'x' only takes numeric values"
        with self.assertRaisesRegex(ValueError, regexp):
            self.XY.x = 'string'

    def test_XY_set_invalid_class_y(self):
        if PARAM_GE_2_0_0:
            regexp = "Number parameter 'XY.y' only takes numeric values"
        else:
            regexp = "Parameter 'y' only takes numeric values"
        with self.assertRaisesRegex(ValueError, regexp):
            self.XY.y = 'string'

    def test_XY_set_invalid_instance_x(self):
        xy = self.XY(x=1,y=2)
        if PARAM_GE_2_0_0:
            regexp = "Number parameter 'XY.x' only takes numeric values"
        else:
            regexp = "Parameter 'x' only takes numeric values"
        with self.assertRaisesRegex(ValueError, regexp):
            xy.x = 'string'

    def test_XY_set_invalid_instance_y(self):
        xy = self.XY(x=1,y=2)
        if PARAM_GE_2_0_0:
            regexp = "Number parameter 'XY.y' only takes numeric values"
        else:
            regexp = "Parameter 'y' only takes numeric values"
        with self.assertRaisesRegex(ValueError, regexp):
            xy.y = 'string'

    def test_XY_subscriber_triggered(self):

        class Inner:
            def __init__(self): self.state=None
            def __call__(self, x,y): self.state=(x,y)

        inner = Inner()
        xy = self.XY(x=1,y=2)
        xy.add_subscriber(inner)
        xy.event(x=42,y=420)
        self.assertEqual(inner.state, (42,420))

    def test_custom_types(self):
        self.assertEqual(isinstance(self.TypesTest.param['t'], param.Boolean),True)
        self.assertEqual(isinstance(self.TypesTest.param['u'], param.Integer),True)
        self.assertEqual(isinstance(self.TypesTest.param['v'], param.Number),True)
        self.assertEqual(isinstance(self.TypesTest.param['w'], param.Tuple),True)
        self.assertEqual(isinstance(self.TypesTest.param['x'], param.String),True)
        self.assertEqual(isinstance(self.TypesTest.param['y'], param.List),True)
        self.assertEqual(isinstance(self.TypesTest.param['z'], param.Array),True)

    def test_explicit_parameter(self):
        self.assertEqual(isinstance(self.ExplicitTest.param['test'], param.Integer),True)
        self.assertEqual(self.ExplicitTest.param['test'].default,42)
        self.assertEqual(self.ExplicitTest.param['test'].doc, 'Test docstring')


class _TestSubscriber:

    def __init__(self, cb=None):
        self.call_count = 0
        self.kwargs = None
        self.cb = cb

    def __call__(self, **kwargs):
        self.call_count += 1
        self.kwargs = kwargs
        if self.cb:
            self.cb()


class TestPointerStreams(ComparisonTestCase):

    def test_positionX_init(self):
        PointerX()

    def test_positionXY_init_contents(self):
        position = PointerXY(x=1, y=3)
        self.assertEqual(position.contents, dict(x=1, y=3))

    def test_positionXY_update_contents(self):
        position = PointerXY()
        position.event(x=5, y=10)
        self.assertEqual(position.contents, dict(x=5, y=10))

    def test_positionY_const_parameter(self):
        position = PointerY()
        try:
            position.y = 5
            raise Exception('No constant parameter exception')
        except TypeError as e:
            self.assertEqual(str(e), "Constant parameter 'y' cannot be modified")



class TestParamsStream(LoggingComparisonTestCase):

    def setUp(self):
        super().setUp()
        class Inner(param.Parameterized):

            x = param.Number(default = 0)
            y = param.Number(default = 0)

        class InnerAction(Inner):

            action = param.Action(default=lambda o: o.param.trigger('action'))

        self.inner = Inner
        self.inner_action = InnerAction

    def test_param_stream_class(self):
        stream = Params(self.inner)
        self.assertEqual(set(stream.parameters), {self.inner.param.x,
                                                  self.inner.param.y})
        self.assertEqual(stream.contents, {'x': 0, 'y': 0})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        self.inner.x = 1
        self.assertEqual(values, [{'x': 1, 'y': 0}])

    def test_param_stream_instance(self):
        inner = self.inner(x=2)
        stream = Params(inner)
        self.assertEqual(set(stream.parameters), {inner.param.x, inner.param.y})
        self.assertEqual(stream.contents, {'x': 2, 'y': 0})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.y = 2
        self.assertEqual(values, [{'x': 2, 'y': 2}])

    def test_param_stream_instance_separate_parameters(self):
        inner = self.inner()

        xparam = Params(inner, ['x'])
        yparam = Params(inner, ['y'])

        valid, invalid = Stream._process_streams([xparam, yparam])
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(invalid), 0)

    def test_param_stream_instance_overlapping_parameters(self):
        inner = self.inner()

        params1 = Params(inner)
        params2 = Params(inner)

        Stream._process_streams([params1, params2])
        self.log_handler.assertContains('WARNING', "['x', 'y']")

    def test_param_parameter_instance_separate_parameters(self):
        inner = self.inner()

        valid, invalid = Stream._process_streams([inner.param.x, inner.param.y])
        xparam, yparam = valid

        self.assertIs(xparam.parameterized, inner)
        self.assertEqual(xparam.parameters, [inner.param.x])
        self.assertIs(yparam.parameterized, inner)
        self.assertEqual(yparam.parameters, [inner.param.y])

    def test_param_parameter_instance_overlapping_parameters(self):
        inner = self.inner()
        Stream._process_streams([inner.param.x, inner.param.x])
        self.log_handler.assertContains('WARNING', "['x']")

    def test_param_stream_parameter_override(self):
        inner = self.inner(x=2)
        stream = Params(inner, parameters=['x'])
        self.assertEqual(stream.parameters, [inner.param.x])
        self.assertEqual(stream.contents, {'x': 2})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.x = 3
        self.assertEqual(values, [{'x': 3}])

    def test_param_stream_rename(self):
        inner = self.inner(x=2)
        stream = Params(inner, rename={'x': 'X', 'y': 'Y'})
        self.assertEqual(set(stream.parameters), {inner.param.x, inner.param.y})
        self.assertEqual(stream.contents, {'X': 2, 'Y': 0})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.y = 2
        self.assertEqual(values, [{'X': 2, 'Y': 2}])

    def test_param_stream_action(self):
        inner = self.inner_action()
        stream = Params(inner, ['action'])
        self.assertEqual(set(stream.parameters), {inner.param.action})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey),
                             {f'{id(inner)} action', '_memoize_key'})

        stream.add_subscriber(subscriber)
        inner.action(inner)
        self.assertEqual(values, [{'action': inner.action}])

    def test_param_stream_memoization(self):
        inner = self.inner_action()
        stream = Params(inner, ['action', 'x'])
        self.assertEqual(set(stream.parameters), {inner.param.action, inner.param.x})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(
                set(stream.hashkey),
                {f'{id(inner)} action', f'{id(inner)} x', '_memoize_key'})

        stream.add_subscriber(subscriber)
        inner.action(inner)
        inner.x = 0
        self.assertEqual(values, [{'action': inner.action, 'x': 0}])

    def test_params_stream_batch_watch(self):
        tap = Tap(x=0, y=1)
        params = Params(parameters=[tap.param.x, tap.param.y])

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
        params.add_subscriber(subscriber)

        tap.param.trigger('x', 'y')

        assert values == [{'x': 0, 'y': 1}]

        tap.event(x=1, y=2)

        assert values == [{'x': 0, 'y': 1}, {'x': 1, 'y': 2}]

    def test_params_no_names(self):
        a = IntSlider()
        b = IntSlider()
        p = Params(parameters=[a.param.value, b.param.value])
        assert len(p.hashkey) == 3  # the two widgets + _memoize_key

    def test_params_identical_names(self):
        a = IntSlider(name="Name")
        b = IntSlider(name="Name")
        p = Params(parameters=[a.param.value, b.param.value])
        assert len(p.hashkey) == 3  # the two widgets + _memoize_key



class TestParamRefsStream(LoggingComparisonTestCase):

    def setUp(self):
        super().setUp()
        class Inner(param.Parameterized):

            x = param.Number(default = 0)
            y = param.Number(default = 0)

        class InnerAction(Inner):

            action = param.Action(default=lambda o: o.param.trigger('action'))

        self.inner = Inner
        self.inner_action = InnerAction

    def test_param_stream_class(self):
        stream = ParamRefs(refs={'x': self.inner.param.x, 'y': self.inner.param.y})
        self.assertEqual(stream.contents, {'x': 0, 'y': 0})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        self.inner.x = 1
        self.assertEqual(values, [{'x': 1, 'y': 0}])

    def test_param_stream_instance(self):
        inner = self.inner(x=2)
        stream = ParamRefs(refs={'x': inner.param.x, 'y': inner.param.y})
        self.assertEqual(stream.contents, {'x': 2, 'y': 0})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.y = 2
        self.assertEqual(values, [{'x': 2, 'y': 2}])
        inner.param.update(x=3, y=3)
        self.assertEqual(values, [{'x': 2, 'y': 2}, {'x': 3, 'y': 3}])

    def test_param_stream_instance_separate_parameters(self):
        inner = self.inner()

        xparam = ParamRefs(refs={'x': inner.param.x})
        yparam = ParamRefs(refs={'y': inner.param.y})

        valid, invalid = Stream._process_streams([xparam, yparam])
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(invalid), 0)

    def test_param_stream_memoization(self):
        inner = self.inner_action()
        stream = ParamRefs(refs={'action': inner.param.action, 'x': inner.param.x})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(
                set(stream.hashkey),
                {'action', 'x', '_memoize_key'})

        stream.add_subscriber(subscriber)
        inner.action(inner)
        inner.x = 0
        self.assertEqual(values, [{'action': inner.action, 'x': 0}])


class TestParamMethodStream(ComparisonTestCase):

    def setUp(self):
        if PARAM_VERSION < (1, 8, 0):
            raise SkipTest('Params stream requires param >= 1.8.0')

        class Inner(param.Parameterized):

            action = param.Action(default=lambda o: o.param.trigger('action'))
            x = param.Number(default = 0)
            y = param.Number(default = 0)
            count = param.Integer(default=0)

            @param.depends('x')
            def method(self):
                self.count += 1
                return Points([])

            @param.depends('action')
            def action_method(self):
                pass

            @param.depends('action', 'x')
            def action_number_method(self):
                self.count += 1
                return Points([])

            @param.depends('y')
            def op_method(self, obj):
                pass

            def method_no_deps(self):
                pass

        class InnerSubObj(Inner):

            sub = param.Parameter()

            @param.depends('sub.x')
            def subobj_method(self):
                pass

        self.inner = Inner
        self.innersubobj = InnerSubObj

    def test_param_method_depends(self):
        inner = self.inner()
        stream = ParamMethod(inner.method)
        self.assertEqual(set(stream.parameters), {inner.param.x})
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.y = 2
        self.assertEqual(values, [{}])

    def test_param_function_depends(self):
        inner = self.inner()

        @param.depends(inner.param.x)
        def test(x):
            return Points([x])

        dmap = DynamicMap(test)

        inner.x = 10
        self.assertEqual(dmap[()], Points([10]))


    def test_param_instance_steams_dict(self):
        inner = self.inner()

        def test(x):
            return Points([x])

        dmap = DynamicMap(test, streams=dict(x=inner.param.x))

        inner.x = 10
        self.assertEqual(dmap[()], Points([10]))

    def test_param_class_steams_dict(self):
        class ClassParamExample(param.Parameterized):
            x = param.Number(default=1)

        def test(x):
            return Points([x])

        dmap = DynamicMap(test, streams=dict(x=ClassParamExample.param.x))

        ClassParamExample.x = 10
        self.assertEqual(dmap[()], Points([10]))

    def test_panel_param_steams_dict(self):
        import panel as pn
        widget = pn.widgets.FloatSlider(value=1)

        def test(x):
            return Points([x])

        dmap = DynamicMap(test, streams=dict(x=widget))

        widget.value = 10
        self.assertEqual(dmap[()], Points([10]))


    def test_param_method_depends_no_deps(self):
        inner = self.inner()
        stream = ParamMethod(inner.method_no_deps)
        self.assertEqual(set(stream.parameters), {
            inner.param.x, inner.param.y, inner.param.action,
            inner.param.name, inner.param.count
        })
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.y = 2
        self.assertEqual(values, [{}, {}])

    def test_param_method_depends_on_subobj(self):
        inner = self.innersubobj(sub=self.inner())
        stream = ParamMethod(inner.subobj_method)
        self.assertEqual(set(stream.parameters), {inner.sub.param.x})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.sub.x = 2
        self.assertEqual(values, [{}])

    def test_dynamicmap_param_method_deps(self):
        inner = self.inner()
        dmap = DynamicMap(inner.method)
        self.assertEqual(len(dmap.streams), 1)
        stream = dmap.streams[0]
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.y = 2
        self.assertEqual(values, [{}])

    def test_param_method_depends_trigger_no_memoization(self):
        inner = self.inner()
        stream = ParamMethod(inner.method)
        self.assertEqual(set(stream.parameters), {inner.param.x})
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.param.trigger('x')
        self.assertEqual(values, [{}, {}])

    def test_dynamicmap_param_method_deps_memoization(self):
        inner = self.inner()
        dmap = DynamicMap(inner.method)
        stream = dmap.streams[0]
        self.assertEqual(set(stream.parameters), {inner.param.x})
        self.assertEqual(stream.contents, {})

        dmap[()]
        dmap[()]
        self.assertEqual(inner.count, 1)

    def test_dynamicmap_param_method_no_deps(self):
        inner = self.inner()
        dmap = DynamicMap(inner.method_no_deps)
        self.assertEqual(dmap.streams, [])

    def test_dynamicmap_param_method_action_param(self):
        inner = self.inner()
        dmap = DynamicMap(inner.action_method)
        self.assertEqual(len(dmap.streams), 1)
        stream = dmap.streams[0]
        self.assertEqual(set(stream.parameters), {inner.param.action})
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey),
                             {f'{id(inner)} action', '_memoize_key'})

        stream.add_subscriber(subscriber)
        inner.action(inner)
        self.assertEqual(values, [{}])

    def test_dynamicmap_param_action_number_method_memoizes(self):
        inner = self.inner()
        dmap = DynamicMap(inner.action_number_method)
        self.assertEqual(len(dmap.streams), 1)
        stream = dmap.streams[0]
        self.assertEqual(set(stream.parameters), {inner.param.action, inner.param.x})
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(
                set(stream.hashkey),
                {f'{id(inner)} action', f'{id(inner)} x', '_memoize_key'})

        stream.add_subscriber(subscriber)
        stream.add_subscriber(lambda **kwargs: dmap[()])
        inner.action(inner)
        self.assertEqual(values, [{}])
        self.assertEqual(inner.count, 1)
        inner.x = 0
        self.assertEqual(values, [{}])
        self.assertEqual(inner.count, 1)

    def test_dynamicmap_param_method_dynamic_operation(self):
        inner = self.inner()
        dmap = DynamicMap(inner.method)
        inner_stream = dmap.streams[0]
        op_dmap = Dynamic(dmap, operation=inner.op_method)
        self.assertEqual(len(op_dmap.streams), 1)
        stream = op_dmap.streams[0]
        self.assertEqual(set(stream.parameters), {inner.param.y})
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})

        values_x, values_y = [], []
        def subscriber_x(**kwargs):
            values_x.append(kwargs)

        def subscriber_y(**kwargs):
            values_y.append(kwargs)

        inner_stream.add_subscriber(subscriber_x)
        stream.add_subscriber(subscriber_y)
        inner.y = 3
        self.assertEqual(values_x, [])
        self.assertEqual(values_y, [{}])


@pytest.mark.usefixtures("bokeh_backend")
def test_dynamicmap_partial_bind_and_streams():
    # Ref: https://github.com/holoviz/holoviews/issues/6008

    def make_plot(z, x_range, y_range):
        return Curve([1, 2, 3, 4, z])

    slider = IntSlider(name='Slider', start=0, end=10)
    range_xy = RangeXY()

    dmap = DynamicMap(param.bind(make_plot, z=slider), streams=[range_xy])

    bk_figure = hv.render(dmap)

    assert bk_figure.renderers[0].data_source.data["y"][-1] == 0
    assert range_xy.x_range == (0, 4)
    assert range_xy.y_range == (-0.4, 4.4)


class TestSubscribers(ComparisonTestCase):

    def test_exception_subscriber(self):
        subscriber = _TestSubscriber()
        position = PointerXY(subscribers=[subscriber])
        kwargs = dict(x=3, y=4)
        position.event(**kwargs)
        self.assertEqual(subscriber.kwargs, kwargs)

    def test_subscriber_disabled(self):
        subscriber = _TestSubscriber()
        position = PointerXY(subscribers=[subscriber])
        kwargs = dict(x=3, y=4)
        position.update(**kwargs)
        self.assertEqual(subscriber.kwargs, None)

    def test_subscribers(self):
        subscriber1 = _TestSubscriber()
        subscriber2 = _TestSubscriber()
        position = PointerXY(subscribers=[subscriber1, subscriber2])
        kwargs = dict(x=3, y=4)
        position.event(**kwargs)
        self.assertEqual(subscriber1.kwargs, kwargs)
        self.assertEqual(subscriber2.kwargs, kwargs)

    def test_batch_subscriber(self):
        subscriber = _TestSubscriber()

        positionX = PointerX(subscribers=[subscriber])
        positionY = PointerY(subscribers=[subscriber])

        positionX.update(x=5)
        positionY.update(y=10)

        Stream.trigger([positionX, positionY])
        self.assertEqual(subscriber.kwargs, dict(x=5, y=10))
        self.assertEqual(subscriber.call_count, 1)

    def test_batch_subscribers(self):
        subscriber1 = _TestSubscriber()
        subscriber2 = _TestSubscriber()

        positionX = PointerX(subscribers=[subscriber1, subscriber2])
        positionY = PointerY(subscribers=[subscriber1, subscriber2])

        positionX.update(x=50)
        positionY.update(y=100)

        Stream.trigger([positionX, positionY])

        self.assertEqual(subscriber1.kwargs, dict(x=50, y=100))
        self.assertEqual(subscriber1.call_count, 1)

        self.assertEqual(subscriber2.kwargs, dict(x=50, y=100))
        self.assertEqual(subscriber2.call_count, 1)

    def test_pipe_memoization(self):
        def points(data):
            subscriber.call_count += 1
            return Points([(0, data)])

        stream = Pipe(data=0)
        dmap = DynamicMap(points, streams=[stream])
        def cb():
            dmap[()]
        subscriber = _TestSubscriber(cb)
        stream.add_subscriber(subscriber)
        dmap[()]
        stream.send(1)

        # Ensure call count was incremented on init, the subscriber
        # and the callback
        self.assertEqual(subscriber.call_count, 3)



class TestStreamSource(ComparisonTestCase):

    def tearDown(self):
        with param.logging_level('ERROR'):
            Stream.registry = defaultdict(list)

    def test_source_empty_element(self):
        points = Points([])
        stream = PointerX(source=points)
        self.assertIs(stream.source, points)

    def test_source_empty_element_remap(self):
        points = Points([])
        stream = PointerX(source=points)
        self.assertIs(stream.source, points)
        curve = Curve([])
        stream.source = curve
        self.assertNotIn(points, Stream.registry)
        self.assertIn(curve, Stream.registry)

    def test_source_empty_dmap(self):
        points_dmap = DynamicMap(lambda x: Points([]), kdims=['X'])
        stream = PointerX(source=points_dmap)
        self.assertIs(stream.source, points_dmap)

    def test_source_registry(self):
        points = Points([(0, 0)])
        PointerX(source=points)
        self.assertIn(points, Stream.registry)

    def test_source_registry_empty_element(self):
        points = Points([])
        PointerX(source=points)
        self.assertIn(points, Stream.registry)


class TestParameterRenaming(ComparisonTestCase):

    def test_simple_rename_constructor(self):
        xy = PointerXY(rename={'x':'xtest', 'y':'ytest'}, x=0, y=4)
        self.assertEqual(xy.contents, {'xtest':0, 'ytest':4})

    def test_invalid_rename_constructor(self):
        regexp = '(.+?)is not a stream parameter'
        with self.assertRaisesRegex(KeyError, regexp):
            PointerXY(rename={'x':'xtest', 'z':'ytest'}, x=0, y=4)
            self.assertEqual(str(cm).endswith(), True)

    def test_clashing_rename_constructor(self):
        regexp = '(.+?)parameter of the same name'
        with self.assertRaisesRegex(KeyError, regexp):
            PointerXY(rename={'x':'xtest', 'y':'x'}, x=0, y=4)

    def test_simple_rename_method(self):
        xy = PointerXY(x=0, y=4)
        renamed = xy.rename(x='xtest', y='ytest')
        self.assertEqual(renamed.contents, {'xtest':0, 'ytest':4})

    def test_invalid_rename_method(self):
        xy = PointerXY(x=0, y=4)
        regexp = '(.+?)is not a stream parameter'
        with self.assertRaisesRegex(KeyError, regexp):
            xy.rename(x='xtest', z='ytest')


    def test_clashing_rename_method(self):
        xy = PointerXY(x=0, y=4)
        regexp = '(.+?)parameter of the same name'
        with self.assertRaisesRegex(KeyError, regexp):
            xy.rename(x='xtest', y='x')

    def test_update_rename_valid(self):
        xy = PointerXY(x=0, y=4)
        renamed = xy.rename(x='xtest', y='ytest')
        renamed.event(x=4, y=8)
        self.assertEqual(renamed.contents, {'xtest':4, 'ytest':8})

    def test_update_rename_invalid(self):
        xy = PointerXY(x=0, y=4)
        renamed = xy.rename(y='ytest')
        regexp = "ytest' is not a parameter of(.+?)"
        with self.assertRaisesRegex(ValueError, regexp):
            renamed.event(ytest=8)

    def test_rename_suppression(self):
        renamed = PointerXY(x=0,y=0).rename(x=None)
        self.assertEqual(renamed.contents, {'y':0})

    def test_rename_suppression_reenable(self):
        renamed = PointerXY(x=0,y=0).rename(x=None)
        self.assertEqual(renamed.contents, {'y':0})
        reenabled = renamed.rename(x='foo')
        self.assertEqual(reenabled.contents, {'foo':0, 'y':0})



class TestPlotSizeTransform(ComparisonTestCase):

    def test_plotsize_initial_contents_1(self):
        plotsize = PlotSize(width=300, height=400, scale=0.5)
        self.assertEqual(plotsize.contents, {'width':300, 'height':400, 'scale':0.5})

    def test_plotsize_update_1(self):
        plotsize = PlotSize(scale=0.5)
        plotsize.event(width=300, height=400)
        self.assertEqual(plotsize.contents, {'width':150, 'height':200, 'scale':0.5})

    def test_plotsize_initial_contents_2(self):
        plotsize = PlotSize(width=600, height=100, scale=2)
        self.assertEqual(plotsize.contents, {'width':600, 'height':100, 'scale':2})

    def test_plotsize_update_2(self):
        plotsize = PlotSize(scale=2)
        plotsize.event(width=600, height=100)
        self.assertEqual(plotsize.contents, {'width':1200, 'height':200, 'scale':2})


class TestPipeStream(ComparisonTestCase):

    def test_pipe_send(self):
        def subscriber(data):
            subscriber.test = data
        subscriber.test = None

        pipe = Pipe()
        pipe.add_subscriber(subscriber)
        pipe.send('Test')
        self.assertEqual(pipe.data, 'Test')
        self.assertEqual(subscriber.test, 'Test')

    def test_pipe_event(self):
        def subscriber(data):
            subscriber.test = data
        subscriber.test = None

        pipe = Pipe()
        pipe.add_subscriber(subscriber)
        pipe.event(data='Test')
        self.assertEqual(pipe.data, 'Test')
        self.assertEqual(subscriber.test, 'Test')

    def test_pipe_update(self):
        pipe = Pipe()
        pipe.event(data='Test')
        self.assertEqual(pipe.data, 'Test')



class TestBufferArrayStream(ComparisonTestCase):

    def test_init_buffer_array(self):
        arr = np.array([[0, 1]])
        buff = Buffer(arr)
        self.assertEqual(buff.data, arr)

    def test_buffer_array_ndim_exception(self):
        error = "Only 2D array data may be streamed by Buffer."
        with self.assertRaisesRegex(ValueError, error):
            Buffer(np.array([0, 1]))

    def test_buffer_array_send(self):
        buff = Buffer(np.array([[0, 1]]))
        buff.send(np.array([[1, 2]]))
        self.assertEqual(buff.data, np.array([[0, 1], [1, 2]]))

    def test_buffer_array_larger_than_length(self):
        buff = Buffer(np.array([[0, 1]]), length=1)
        buff.send(np.array([[1, 2]]))
        self.assertEqual(buff.data, np.array([[1, 2]]))

    def test_buffer_array_patch_larger_than_length(self):
        buff = Buffer(np.array([[0, 1]]), length=1)
        buff.send(np.array([[1, 2], [2, 3]]))
        self.assertEqual(buff.data, np.array([[2, 3]]))

    def test_buffer_array_send_verify_ndim_fail(self):
        buff = Buffer(np.array([[0, 1]]))
        error = 'Streamed array data must be two-dimensional'
        with self.assertRaisesRegex(ValueError, error):
            buff.send(np.array([1]))

    def test_buffer_array_send_verify_shape_fail(self):
        buff = Buffer(np.array([[0, 1]]))
        error = "Streamed array data expected to have 2 columns, got 3."
        with self.assertRaisesRegex(ValueError, error):
            buff.send(np.array([[1, 2, 3]]))

    def test_buffer_array_send_verify_type_fail(self):
        buff = Buffer(np.array([[0, 1]]))
        error = "Input expected to be of type ndarray, got list."
        with self.assertRaisesRegex(TypeError, error):
            buff.send([1])


class TestBufferDictionaryStream(ComparisonTestCase):

    def test_init_buffer_dict(self):
        data = {'x': np.array([1]), 'y': np.array([2])}
        buff = Buffer(data)
        self.assertEqual(buff.data, data)

    def test_buffer_dict_send(self):
        data = {'x': np.array([0]), 'y': np.array([1])}
        buff = Buffer(data)
        buff.send({'x': np.array([1]), 'y': np.array([2])})
        self.assertEqual(buff.data, {'x': np.array([0, 1]), 'y': np.array([1, 2])})

    def test_buffer_dict_larger_than_length(self):
        data = {'x': np.array([0]), 'y': np.array([1])}
        buff = Buffer(data, length=1)
        chunk = {'x': np.array([1]), 'y': np.array([2])}
        buff.send(chunk)
        self.assertEqual(buff.data, chunk)

    def test_buffer_dict_patch_larger_than_length(self):
        data = {'x': np.array([0]), 'y': np.array([1])}
        buff = Buffer(data, length=1)
        chunk = {'x': np.array([1, 2]), 'y': np.array([2, 3])}
        buff.send(chunk)
        self.assertEqual(buff.data, {'x': np.array([2]), 'y': np.array([3])})

    def test_buffer_dict_send_verify_column_fail(self):
        data = {'x': np.array([0]), 'y': np.array([1])}
        buff = Buffer(data)
        error = r"Input expected to have columns \['x', 'y'\], got \['x'\]"
        with self.assertRaisesRegex(IndexError, error):
            buff.send({'x': np.array([2])})

    def test_buffer_dict_send_verify_shape_fail(self):
        data = {'x': np.array([0]), 'y': np.array([1])}
        buff = Buffer(data)
        error = "Input columns expected to have the same number of rows."
        with self.assertRaisesRegex(ValueError, error):
            buff.send({'x': np.array([2]), 'y': np.array([3, 4])})


class TestBufferDataFrameStream(ComparisonTestCase):

    def test_init_buffer_dframe(self):
        data = pd.DataFrame({'x': np.array([1]), 'y': np.array([2])})
        buff = Buffer(data, index=False)
        self.assertEqual(buff.data, data)

    def test_init_buffer_dframe_with_index(self):
        data = pd.DataFrame({'x': np.array([1]), 'y': np.array([2])})
        buff = Buffer(data)
        self.assertEqual(buff.data, data)

    def test_buffer_dframe_send(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data, index=False)
        buff.send(pd.DataFrame({'x': np.array([1]), 'y': np.array([2])}))
        dframe = pd.DataFrame({'x': np.array([0, 1]), 'y': np.array([1, 2])})
        self.assertEqual(buff.data.values, dframe.values)

    def test_buffer_dframe_send_with_index(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data)
        buff.send(pd.DataFrame({'x': np.array([1]), 'y': np.array([2])}))
        dframe = pd.DataFrame({'x': np.array([0, 1]), 'y': np.array([1, 2])}, index=[0, 0])
        pd.testing.assert_frame_equal(buff.data, dframe)

    def test_buffer_dframe_larger_than_length(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data, length=1, index=False)
        chunk = pd.DataFrame({'x': np.array([1]), 'y': np.array([2])})
        buff.send(chunk)
        self.assertEqual(buff.data.values, chunk.values)

    def test_buffer_dframe_patch_larger_than_length(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data, length=1, index=False)
        chunk = pd.DataFrame({'x': np.array([1, 2]), 'y': np.array([2, 3])})
        buff.send(chunk)
        dframe = pd.DataFrame({'x': np.array([2]), 'y': np.array([3])})
        self.assertEqual(buff.data.values, dframe.values)

    def test_buffer_dframe_send_verify_column_fail(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data, index=False)
        error = r"Input expected to have columns \['x', 'y'\], got \['x'\]"
        with self.assertRaisesRegex(IndexError, error):
            buff.send(pd.DataFrame({'x': np.array([2])}))

    def test_clear_buffer_dframe_with_index(self):
        data = pd.DataFrame({'a': [1, 2, 3]})
        buff = Buffer(data)
        buff.clear()
        pd.testing.assert_frame_equal(buff.data, data.iloc[:0, :])


class Sum(Derived):
    v = param.Number(constant=True)

    def __init__(self, val_streams, exclusive=False, base=0):
        self.base = base
        super().__init__(input_streams=val_streams, exclusive=exclusive)

    @property
    def constants(self):
        return dict(base=self.base)

    @classmethod
    def transform_function(cls, stream_values, constants):
        v = sum([val["v"] for val in stream_values if val["v"]])
        return dict(v=v + constants['base'])


Val = Stream.define("Val", v=0.0)


class TestDerivedStream(ComparisonTestCase):

    def test_simple_derived_stream(self):
        # Define input streams
        v0 = Val(v=1.0)
        v1 = Val(v=2.0)

        # Build Sum stream
        s0 = Sum([v0, v1])

        # Check outputs
        self.assertEqual(s0.v, 3.0)

        # Update v0
        v0.event(v=7.0)
        self.assertEqual(s0.v, 9.0)

        # Update v1
        v1.event(v=-8.0)
        self.assertEqual(s0.v, -1.0)

    def test_nested_derived_stream(self):
        v0 = Val(v=1.0)
        v1 = Val(v=4.0)
        v2 = Val(v=7.0)

        # Build nested sum stream
        s1 = Sum([v0, v1])
        s0 = Sum([s1, v2])

        # Check initial value
        self.assertEqual(s0.v, 12.0)

        # Update top-level value
        v2.event(v=8.0)
        self.assertEqual(s0.v, 13.0)

        # Update nested value
        v1.event(v=5.0,)
        self.assertEqual(s0.v, 14.0)

    def test_derived_stream_constants(self):
        v0 = Val(v=1.0)
        v1 = Val(v=4.0)
        v2 = Val(v=7.0)

        # Build Sum stream with base value
        s0 = Sum([v0, v1, v2], base=100)

        # Check initial value
        self.assertEqual(s0.v, 112.0)

        # Update value
        v2.event(v=8.0)
        self.assertEqual(s0.v, 113.0)

    def test_exclusive_derived_stream(self):
        # Define input streams
        v0 = Val()
        v1 = Val(v=2.0)

        # Build exclusive Sum stream
        # In this case, all streams except the most recently updated will be reset on
        # update
        s0 = Sum([v0, v1], exclusive=True)

        # Check outputs
        self.assertEqual(s0.v, 2.0)

        # Update v0
        v0.event(v=7.0)
        self.assertEqual(s0.v, 7.0)

        # Update v1
        v1.event(v=-8.0)
        self.assertEqual(s0.v, -8.0)


class TestHistoryStream(ComparisonTestCase):
    def test_initial_history_stream_values(self):
        # Check values list is initialized with initial contents of input stream
        val = Val(v=1.0)
        history = History(val)
        self.assertEqual(history.contents, {"values": [val.contents]})

    def test_history_stream_values_appended(self):
        val = Val(v=1.0)
        history = History(val)
        # Perform a few updates on val stream
        val.event(v=2.0)
        val.event(v=3.0)
        self.assertEqual(
            history.contents,
            {"values": [{"v": 1.0}, {"v": 2.0}, {"v": 3.0}]}
        )

        # clear
        history.clear_history()
        self.assertEqual(history.contents, {"values": []})

    def test_history_stream_trigger_callbacks(self):
        # Construct history stream
        val = Val(v=1.0)
        history = History(val)

        # Register callback
        callback_input = []
        def cb(**kwargs):
            callback_input.append(kwargs)
        history.add_subscriber(cb)
        self.assertEqual(callback_input, [])

        # Perform updates on val stream and make sure history callback is triggered
        del callback_input[:]
        val.event(v=2.0)
        self.assertEqual(
            callback_input[0],
            {"values": [{"v": 1.0}, {"v": 2.0}]}
        )

        del callback_input[:]
        val.event(v=3.0)
        self.assertEqual(
            callback_input[0],
            {"values": [{"v": 1.0}, {"v": 2.0}, {"v": 3.0}]}
        )

        # clearing history should trigger callback
        del callback_input[:]
        history.clear_history()
        history.event()
        self.assertEqual(
            callback_input[0],
            {"values": []}
        )

        # Update after clearing
        del callback_input[:]
        val.event(v=4.0)
        self.assertEqual(
            callback_input[0],
            {"values": [{"v": 4.0}]}
        )


class TestExprSelectionStream(ComparisonTestCase):

    def setUp(self):
        extension("bokeh")

    def test_selection_expr_stream_2D_elements(self):
        element_type_2D = [Points]
        for element_type in element_type_2D:
            # Create SelectionExpr on element
            element = element_type(([1, 2, 3], [1, 5, 10]))
            expr_stream = SelectionExpr(element)

            # Check stream properties
            self.assertEqual(len(expr_stream.input_streams), 3)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsInstance(expr_stream.input_streams[1], Lasso)
            self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)

            # Simulate interactive update by triggering source stream
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))

            # Check SelectionExpr values
            self.assertEqual(
                repr(expr_stream.selection_expr),
                repr(((dim('x')>=1)&(dim('x')<=3))&((dim('y')>=1)&(dim('y')<=4)))
            )
            self.assertEqual(
                expr_stream.bbox,
                {'x': (1, 3), 'y': (1, 4)}
            )

    def test_selection_expr_stream_1D_elements(self):
        element_type_1D = [Scatter]
        for element_type in element_type_1D:
            # Create SelectionExpr on element
            element = element_type(([1, 2, 3], [1, 5, 10]))
            expr_stream = SelectionExpr(element)

            # Check stream properties
            self.assertEqual(len(expr_stream.input_streams), 1)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)

            # Simulate interactive update by triggering source stream
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))

            # Check SelectionExpr values
            self.assertEqual(
                repr(expr_stream.selection_expr),
                repr((dim('x')>=1)&(dim('x')<=3))
            )
            self.assertEqual(
                expr_stream.bbox,
                {'x': (1, 3)}
            )


    def test_selection_expr_stream_invert_axes_2D_elements(self):
        element_type_2D = [Points]
        for element_type in element_type_2D:
            # Create SelectionExpr on element
            element = element_type(([1, 2, 3], [1, 5, 10])).opts(invert_axes=True)
            expr_stream = SelectionExpr(element)

            # Check stream properties
            self.assertEqual(len(expr_stream.input_streams), 3)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsInstance(expr_stream.input_streams[1], Lasso)
            self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)

            # Simulate interactive update by triggering source stream
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))

            # Check SelectionExpr values
            self.assertEqual(
                repr(expr_stream.selection_expr),
                repr(((dim('y')>=1)&(dim('y')<=3))&((dim('x')>=1)&(dim('x')<=4)))
            )
            self.assertEqual(
                expr_stream.bbox,
                {'y': (1, 3), 'x': (1, 4)}
            )

    def test_selection_expr_stream_invert_axes_1D_elements(self):
        element_type_1D = [Scatter]
        for element_type in element_type_1D:
            # Create SelectionExpr on element
            element = element_type(([1, 2, 3], [1, 5, 10])).opts(invert_axes=True)
            expr_stream = SelectionExpr(element)

            # Check stream properties
            self.assertEqual(len(expr_stream.input_streams), 1)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)

            # Simulate interactive update by triggering source stream
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))

            # Check SelectionExpr values
            self.assertEqual(
                repr(expr_stream.selection_expr),
                repr((dim('x')>=1)&(dim('x')<=4))
            )
            self.assertEqual(
                expr_stream.bbox,
                {'x': (1, 4)}
            )

    def test_selection_expr_stream_invert_xaxis_yaxis_2D_elements(self):
        element_type_2D = [Points]
        for element_type in element_type_2D:

            # Create SelectionExpr on element
            element = element_type(([1, 2, 3], [1, 5, 10])).opts(
                invert_xaxis=True,
                invert_yaxis=True,
            )
            expr_stream = SelectionExpr(element)

            # Check stream properties
            self.assertEqual(len(expr_stream.input_streams), 3)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsInstance(expr_stream.input_streams[1], Lasso)
            self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)

            # Simulate interactive update by triggering source stream
            expr_stream.input_streams[0].event(bounds=(3, 4, 1, 1))

            # Check SelectionExpr values
            self.assertEqual(
                repr(expr_stream.selection_expr),
                repr(((dim('x')>=1)&(dim('x')<=3))&((dim('y')>=1)&(dim('y')<=4)))
            )
            self.assertEqual(
                expr_stream.bbox,
                {'x': (1, 3), 'y': (1, 4)}
            )

    def test_selection_expr_stream_invert_xaxis_yaxis_1D_elements(self):
        element_type_1D = [Scatter]
        for element_type in element_type_1D:

            # Create SelectionExpr on element
            element = element_type(([1, 2, 3], [1, 5, 10])).opts(
                invert_xaxis=True,
                invert_yaxis=True,
            )
            expr_stream = SelectionExpr(element)

            # Check stream properties
            self.assertEqual(len(expr_stream.input_streams), 1)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)

            # Simulate interactive update by triggering source stream
            expr_stream.input_streams[0].event(bounds=(3, 4, 1, 1))

            # Check SelectionExpr values
            self.assertEqual(
                repr(expr_stream.selection_expr),
                repr((dim('x')>=1)&(dim('x')<=3))
            )
            self.assertEqual(
                expr_stream.bbox,
                {'x': (1, 3)}
            )

    def test_selection_expr_stream_hist(self):
        # Create SelectionExpr on element
        hist = Histogram(([1, 2, 3, 4, 5], [1, 5, 2, 3, 7]))
        expr_stream = SelectionExpr(hist)

        # Check stream properties
        self.assertEqual(len(expr_stream.input_streams), 1)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)

        # Simulate interactive update by triggering source stream.
        # Select second and forth bar.
        expr_stream.input_streams[0].event(bounds=(1.5, 2.5, 4.6, 6))
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr((dim('x')>=1.5)&(dim('x')<=4.6))
        )
        self.assertEqual(expr_stream.bbox, {'x': (1.5, 4.6)})

        # Select third, forth, and fifth bar.  Make sure there is special
        # handling when last bar is selected to include values exactly on the
        # upper edge in the selection
        expr_stream.input_streams[0].event(bounds=(2.5, -10, 8, 10))
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr((dim('x')>=2.5)&(dim('x')<=8))
        )
        self.assertEqual(expr_stream.bbox, {'x': (2.5, 8)})

    def test_selection_expr_stream_hist_invert_axes(self):
        # Create SelectionExpr on element
        hist = Histogram(([1, 2, 3, 4, 5], [1, 5, 2, 3, 7])).opts(
            invert_axes=True
        )
        expr_stream = SelectionExpr(hist)

        # Check stream properties
        self.assertEqual(len(expr_stream.input_streams), 1)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)

        # Simulate interactive update by triggering source stream.
        # Select second and forth bar.
        expr_stream.input_streams[0].event(bounds=(2.5, 1.5, 6, 4.6))
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr((dim('x')>=1.5)&(dim('x')<=4.6))
        )
        self.assertEqual(expr_stream.bbox, {'x': (1.5, 4.6)})

        # Select third, forth, and fifth bar.  Make sure there is special
        # handling when last bar is selected to include values exactly on the
        # upper edge in the selection
        expr_stream.input_streams[0].event(bounds=(-10, 2.5, 10, 8))
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr((dim('x')>=2.5)&(dim('x')<=8))
        )
        self.assertEqual(expr_stream.bbox, {'x': (2.5, 8)})

    def test_selection_expr_stream_hist_invert_xaxis_yaxis(self):
        # Create SelectionExpr on element
        hist = Histogram(([1, 2, 3, 4, 5], [1, 5, 2, 3, 7])).opts(
                invert_xaxis=True,
                invert_yaxis=True,
        )
        expr_stream = SelectionExpr(hist)

        # Check stream properties
        self.assertEqual(len(expr_stream.input_streams), 1)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)

        # Simulate interactive update by triggering source stream.
        # Select second and forth bar.
        expr_stream.input_streams[0].event(bounds=(4.6, 6, 1.5, 2.5))
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr((dim('x')>=1.5)&(dim('x')<=4.6))
        )
        self.assertEqual(expr_stream.bbox, {'x': (1.5, 4.6)})

        # Select third, forth, and fifth bar.  Make sure there is special
        # handling when last bar is selected to include values exactly on the
        # upper edge in the selection
        expr_stream.input_streams[0].event(bounds=(8, 10, 2.5, -10))
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr((dim('x')>=2.5)&(dim('x')<=8))
        )
        self.assertEqual(expr_stream.bbox, {'x': (2.5, 8)})


    def test_selection_expr_stream_polygon_index_cols(self):
        # TODO: Should test both spatialpandas and shapely
        # Create SelectionExpr on element
        try: import shapely # noqa
        except ImportError:
            try: import spatialpandas # noqa
            except ImportError: raise SkipTest('Shapely required for polygon selection')
        poly = Polygons([
            [(0, 0, 'a'), (2, 0, 'a'), (1, 1, 'a')],
            [(2, 0, 'b'), (4, 0, 'b'), (3, 1, 'b')],
            [(1, 1, 'c'), (3, 1, 'c'), (2, 2, 'c')]
        ], vdims=['cat'])

        events = []
        expr_stream = SelectionExpr(poly, index_cols=['cat'])
        expr_stream.add_subscriber(lambda **kwargs: events.append(kwargs))

        # Check stream properties
        self.assertEqual(len(expr_stream.input_streams), 3)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsInstance(expr_stream.input_streams[1], Lasso)
        self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)

        fmt = lambda x: list(map(np.str_, x)) if NUMPY_GE_2_0_0 else x

        expr_stream.input_streams[2].event(index=[0, 1])
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr(dim('cat').isin(fmt(['a', 'b'])))
        )
        self.assertEqual(expr_stream.bbox, None)
        self.assertEqual(len(events), 1)

        # Ensure bounds event does not trigger another update
        expr_stream.input_streams[0].event(bounds=(0, 0, 4, 1))
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr(dim('cat').isin(fmt(['a', 'b'])))
        )
        self.assertEqual(len(events), 1)

        # Ensure geometry event does trigger another update
        expr_stream.input_streams[1].event(geometry=np.array([(0, 0), (4, 0), (4, 2), (0, 2)]))
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr(dim('cat').isin(fmt(['a', 'b', 'c'])))
        )
        self.assertEqual(len(events), 2)

        # Ensure index event does trigger another update
        expr_stream.input_streams[2].event(index=[1, 2])
        self.assertEqual(
            repr(expr_stream.selection_expr),
            repr(dim('cat').isin(fmt(['b', 'c'])))
        )
        self.assertEqual(expr_stream.bbox, None)
        self.assertEqual(len(events), 3)

    def test_selection_expr_stream_dynamic_map_2D_elements(self):
        element_type_2D = [Points]
        for element_type in element_type_2D: # Scatter,
            # Create SelectionExpr on element
            dmap = Dynamic(element_type(([1, 2, 3], [1, 5, 10])))
            expr_stream = SelectionExpr(dmap)

            # Check stream properties
            self.assertEqual(len(expr_stream.input_streams), 3)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)

            # Simulate interactive update by triggering source stream
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))

            # Check SelectionExpr values
            self.assertEqual(
                repr(expr_stream.selection_expr),
                repr(((dim('x')>=1)&(dim('x')<=3))&((dim('y')>=1)&(dim('y')<=4)))
            )
            self.assertEqual(
                expr_stream.bbox,
                {'x': (1, 3), 'y': (1, 4)}
            )

    def test_selection_expr_stream_dynamic_map_1D_elements(self):
        element_type_1D = [Scatter]
        for element_type in element_type_1D:
            # Create SelectionExpr on element
            dmap = Dynamic(element_type(([1, 2, 3], [1, 5, 10])))
            expr_stream = SelectionExpr(dmap)

            # Check stream properties
            self.assertEqual(len(expr_stream.input_streams), 1)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)

            # Simulate interactive update by triggering source stream
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))

            # Check SelectionExpr values
            self.assertEqual(
                repr(expr_stream.selection_expr),
                repr((dim('x')>=1)&(dim('x')<=3))
            )
            self.assertEqual(
                expr_stream.bbox,
                {'x': (1, 3)}
            )
