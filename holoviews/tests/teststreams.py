"""
Unit test of the streams system
"""
from collections import defaultdict
from unittest import SkipTest

import param
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import LooseVersion, pd
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import * # noqa (Test all available streams)
from holoviews.util import Dynamic

def test_all_stream_parameters_constant():
    all_stream_cls = [v for v in globals().values() if
                      isinstance(v, type) and issubclass(v, Stream)]
    for stream_cls in all_stream_cls:
        for name, p in stream_cls.params().items():
            if name == 'name': continue
            if p.constant != True:
                raise TypeError('Parameter %s of stream %s not declared constant'
                                % (name, stream_cls.__name__))


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
        self.assertEqual(isinstance(self.XY.params('x'), param.Number),True)
        self.assertEqual(isinstance(self.XY.params('y'), param.Number),True)

    def test_XY_defaults(self):
        self.assertEqual(self.XY.params('x').default,0.0)
        self.assertEqual(self.XY.params('y').default, 5.0)

    def test_XY_instance(self):
        xy = self.XY(x=1,y=2)
        self.assertEqual(xy.x, 1)
        self.assertEqual(xy.y, 2)

    def test_XY_set_invalid_class_x(self):
        regexp = "Parameter 'x' only takes numeric values"
        with self.assertRaisesRegexp(ValueError, regexp):
            self.XY.x = 'string'

    def test_XY_set_invalid_class_y(self):
        regexp = "Parameter 'y' only takes numeric values"
        with self.assertRaisesRegexp(ValueError, regexp):
            self.XY.y = 'string'

    def test_XY_set_invalid_instance_x(self):
        xy = self.XY(x=1,y=2)
        regexp = "Parameter 'x' only takes numeric values"
        with self.assertRaisesRegexp(ValueError, regexp):
            xy.x = 'string'

    def test_XY_set_invalid_instance_y(self):
        xy = self.XY(x=1,y=2)
        regexp = "Parameter 'y' only takes numeric values"
        with self.assertRaisesRegexp(ValueError, regexp):
            xy.y = 'string'

    def test_XY_subscriber_triggered(self):

        class Inner(object):
            def __init__(self): self.state=None
            def __call__(self, x,y): self.state=(x,y)

        inner = Inner()
        xy = self.XY(x=1,y=2)
        xy.add_subscriber(inner)
        xy.event(x=42,y=420)
        self.assertEqual(inner.state, (42,420))

    def test_custom_types(self):
        self.assertEqual(isinstance(self.TypesTest.params('t'), param.Boolean),True)
        self.assertEqual(isinstance(self.TypesTest.params('u'), param.Integer),True)
        self.assertEqual(isinstance(self.TypesTest.params('v'), param.Number),True)
        self.assertEqual(isinstance(self.TypesTest.params('w'), param.Tuple),True)
        self.assertEqual(isinstance(self.TypesTest.params('x'), param.String),True)
        self.assertEqual(isinstance(self.TypesTest.params('y'), param.List),True)
        self.assertEqual(isinstance(self.TypesTest.params('z'), param.Array),True)

    def test_explicit_parameter(self):
        self.assertEqual(isinstance(self.ExplicitTest.params('test'), param.Integer),True)
        self.assertEqual(self.ExplicitTest.params('test').default,42)
        self.assertEqual(self.ExplicitTest.params('test').doc, 'Test docstring')


class TestSubscriber(object):

    def __init__(self):
        self.call_count = 0
        self.kwargs = None

    def __call__(self, **kwargs):
        self.call_count += 1
        self.kwargs = kwargs


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


class TestParamValuesStream(ComparisonTestCase):

    def setUp(self):

        class Inner(param.Parameterized):

            x = param.Number(default = 0)
            y = param.Number(default = 0)

        self.inner = Inner

    def tearDown(self):
        self.inner.x = 0
        self.inner.y = 0

    def test_object_contents(self):
        obj = self.inner()
        stream = ParamValues(obj)
        self.assertEqual(stream.contents, {'x':0, 'y':0})

    def test_class_value(self):
        stream = ParamValues(self.inner)
        self.assertEqual(stream.contents, {'x':0, 'y':0})

    def test_object_value_update(self):
        obj = self.inner()
        stream = ParamValues(obj)
        self.assertEqual(stream.contents, {'x':0, 'y':0})
        stream.event(x=5, y=10)
        self.assertEqual(stream.contents, {'x':5, 'y':10})

    def test_class_value_update(self):
        stream = ParamValues(self.inner)
        self.assertEqual(stream.contents, {'x':0, 'y':0})
        stream.event(x=5, y=10)
        self.assertEqual(stream.contents, {'x':5, 'y':10})



class TestParamsStream(ComparisonTestCase):

    def setUp(self):
        if LooseVersion(param.__version__) < '1.8.0':
            raise SkipTest('Params stream requires param >= 1.8.0')

        class Inner(param.Parameterized):

            x = param.Number(default = 0)
            y = param.Number(default = 0)

        class InnerAction(Inner):

            action = param.Action(lambda o: o.param.trigger('action'))

        self.inner = Inner
        self.inner_action = InnerAction

    def test_param_stream_class(self):
        stream = Params(self.inner)
        self.assertEqual(set(stream.parameters), {'x', 'y'})
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
        self.assertEqual(set(stream.parameters), {'x', 'y'})
        self.assertEqual(stream.contents, {'x': 2, 'y': 0})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.y = 2
        self.assertEqual(values, [{'x': 2, 'y': 2}])

    def test_param_stream_parameter_override(self):
        inner = self.inner(x=2)
        stream = Params(inner, parameters=['x'])
        self.assertEqual(stream.parameters, ['x'])
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
        self.assertEqual(set(stream.parameters), {'x', 'y'})
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
        self.assertEqual(set(stream.parameters), {'action'})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey),
                             {'action', '_memoize_key'})

        stream.add_subscriber(subscriber)
        inner.action(inner)
        self.assertEqual(values, [{'action': inner.action}])

    def test_param_stream_memoization(self):
        inner = self.inner_action()
        stream = Params(inner, ['action', 'x'])
        self.assertEqual(set(stream.parameters), {'action', 'x'})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey),
                             {'action', 'x', '_memoize_key'})

        stream.add_subscriber(subscriber)
        inner.action(inner)
        inner.x = 0
        self.assertEqual(values, [{'action': inner.action, 'x': 0}])

        

class TestParamMethodStream(ComparisonTestCase):

    def setUp(self):
        if LooseVersion(param.__version__) < '1.8.0':
            raise SkipTest('Params stream requires param >= 1.8.0')

        class Inner(param.Parameterized):

            action = param.Action(lambda o: o.param.trigger('action'))
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

        self.inner = Inner

    def test_param_method_depends(self):
        inner = self.inner()
        stream = ParamMethod(inner.method)
        self.assertEqual(set(stream.parameters), {'x'})
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.y = 2
        self.assertEqual(values, [{}])

    def test_param_method_depends_no_deps(self):
        inner = self.inner()
        stream = ParamMethod(inner.method_no_deps)
        self.assertEqual(set(stream.parameters), {'x', 'y', 'action', 'name', 'count'})
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)

        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.y = 2
        self.assertEqual(values, [{}, {}])

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
        self.assertEqual(set(stream.parameters), {'x'})
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
        self.assertEqual(set(stream.parameters), {'x'})
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
        self.assertEqual(set(stream.parameters), {'action'})
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey),
                             {'action', '_memoize_key'})

        stream.add_subscriber(subscriber)
        inner.action(inner)
        self.assertEqual(values, [{}])

    def test_dynamicmap_param_action_number_method_memoizes(self):
        inner = self.inner()
        dmap = DynamicMap(inner.action_number_method)
        self.assertEqual(len(dmap.streams), 1)
        stream = dmap.streams[0]
        self.assertEqual(set(stream.parameters), {'action', 'x'})
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})

        values = []
        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey),
                             {'action', 'x', '_memoize_key'})

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
        self.assertEqual(set(stream.parameters), {'y'})
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


        
class TestSubscribers(ComparisonTestCase):

    def test_exception_subscriber(self):
        subscriber = TestSubscriber()
        position = PointerXY(subscribers=[subscriber])
        kwargs = dict(x=3, y=4)
        position.event(**kwargs)
        self.assertEqual(subscriber.kwargs, kwargs)

    def test_subscriber_disabled(self):
        subscriber = TestSubscriber()
        position = PointerXY(subscribers=[subscriber])
        kwargs = dict(x=3, y=4)
        position.update(**kwargs)
        self.assertEqual(subscriber.kwargs, None)


    def test_subscribers(self):
        subscriber1 = TestSubscriber()
        subscriber2 = TestSubscriber()
        position = PointerXY(subscribers=[subscriber1, subscriber2])
        kwargs = dict(x=3, y=4)
        position.event(**kwargs)
        self.assertEqual(subscriber1.kwargs, kwargs)
        self.assertEqual(subscriber2.kwargs, kwargs)

    def test_batch_subscriber(self):
        subscriber = TestSubscriber()

        positionX = PointerX(subscribers=[subscriber])
        positionY = PointerY(subscribers=[subscriber])

        positionX.update(x=5)
        positionY.update(y=10)

        Stream.trigger([positionX, positionY])
        self.assertEqual(subscriber.kwargs, dict(x=5, y=10))
        self.assertEqual(subscriber.call_count, 1)

    def test_batch_subscribers(self):
        subscriber1 = TestSubscriber()
        subscriber2 = TestSubscriber()

        positionX = PointerX(subscribers=[subscriber1, subscriber2])
        positionY = PointerY(subscribers=[subscriber1, subscriber2])

        positionX.update(x=50)
        positionY.update(y=100)

        Stream.trigger([positionX, positionY])

        self.assertEqual(subscriber1.kwargs, dict(x=50, y=100))
        self.assertEqual(subscriber1.call_count, 1)

        self.assertEqual(subscriber2.kwargs, dict(x=50, y=100))
        self.assertEqual(subscriber2.call_count, 1)


class TestStreamSource(ComparisonTestCase):

    def tearDown(self):
        Stream.registry = defaultdict(list)

    def test_source_empty_element(self):
        points = Points([])
        stream = PointerX(source=points)
        self.assertIs(stream.source, points)

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
        with self.assertRaisesRegexp(KeyError, regexp):
            PointerXY(rename={'x':'xtest', 'z':'ytest'}, x=0, y=4)
            self.assertEqual(str(cm).endswith(), True)

    def test_clashing_rename_constructor(self):
        regexp = '(.+?)parameter of the same name'
        with self.assertRaisesRegexp(KeyError, regexp):
            PointerXY(rename={'x':'xtest', 'y':'x'}, x=0, y=4)

    def test_simple_rename_method(self):
        xy = PointerXY(x=0, y=4)
        renamed = xy.rename(x='xtest', y='ytest')
        self.assertEqual(renamed.contents, {'xtest':0, 'ytest':4})

    def test_invalid_rename_method(self):
        xy = PointerXY(x=0, y=4)
        regexp = '(.+?)is not a stream parameter'
        with self.assertRaisesRegexp(KeyError, regexp):
            xy.rename(x='xtest', z='ytest')


    def test_clashing_rename_method(self):
        xy = PointerXY(x=0, y=4)
        regexp = '(.+?)parameter of the same name'
        with self.assertRaisesRegexp(KeyError, regexp):
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
        with self.assertRaisesRegexp(ValueError, regexp):
            renamed.event(ytest=8)

    def test_rename_suppression(self):
        renamed = PointerXY(x=0,y=0).rename(x=None)
        self.assertEquals(renamed.contents, {'y':0})

    def test_rename_suppression_reenable(self):
        renamed = PointerXY(x=0,y=0).rename(x=None)
        self.assertEquals(renamed.contents, {'y':0})
        reenabled = renamed.rename(x='foo')
        self.assertEquals(reenabled.contents, {'foo':0, 'y':0})


        
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
        test = None
        def subscriber(data):
            global test
            test = data
        pipe = Pipe()
        pipe.add_subscriber(subscriber)
        pipe.send('Test')
        self.assertEqual(pipe.data, 'Test')

    def test_pipe_event(self):
        test = None
        def subscriber(data):
            global test
            test = data
        pipe = Pipe()
        pipe.add_subscriber(subscriber)
        pipe.event(data='Test')
        self.assertEqual(pipe.data, 'Test')

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
        with self.assertRaisesRegexp(ValueError, error):
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
        with self.assertRaisesRegexp(ValueError, error):
            buff.send(np.array([1]))

    def test_buffer_array_send_verify_shape_fail(self):
        buff = Buffer(np.array([[0, 1]]))
        error = "Streamed array data expeced to have 2 columns, got 3."
        with self.assertRaisesRegexp(ValueError, error):
            buff.send(np.array([[1, 2, 3]]))

    def test_buffer_array_send_verify_type_fail(self):
        buff = Buffer(np.array([[0, 1]]))
        error = "Input expected to be of type ndarray, got list."
        with self.assertRaisesRegexp(TypeError, error):
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
        error = "Input expected to have columns \['x', 'y'\], got \['x'\]"
        with self.assertRaisesRegexp(IndexError, error):
            buff.send({'x': np.array([2])})

    def test_buffer_dict_send_verify_shape_fail(self):
        data = {'x': np.array([0]), 'y': np.array([1])}
        buff = Buffer(data)
        error = "Input columns expected to have the same number of rows."
        with self.assertRaisesRegexp(ValueError, error):
            buff.send({'x': np.array([2]), 'y': np.array([3, 4])})


class TestBufferDataFrameStream(ComparisonTestCase):

    def setUp(self):
        if pd is None:
            raise SkipTest('Pandas not available')
        super(TestBufferDataFrameStream, self).setUp()

    def test_init_buffer_dframe(self):
        data = pd.DataFrame({'x': np.array([1]), 'y': np.array([2])})
        buff = Buffer(data, index=False)
        self.assertEqual(buff.data, data)

    def test_init_buffer_dframe_with_index(self):
        data = pd.DataFrame({'x': np.array([1]), 'y': np.array([2])})
        buff = Buffer(data)
        self.assertEqual(buff.data, data.reset_index())

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
        self.assertEqual(buff.data.values, dframe.reset_index().values)

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
        error = "Input expected to have columns \['x', 'y'\], got \['x'\]"
        with self.assertRaisesRegexp(IndexError, error):
            buff.send(pd.DataFrame({'x': np.array([2])}))

    def test_clear_buffer_dframe_with_index(self):
        data = pd.DataFrame({'a': [1, 2, 3]})
        buff = Buffer(data)
        buff.clear()
        self.assertEqual(buff.data, data.iloc[:0, :].reset_index())
