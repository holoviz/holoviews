import uuid
import time
from collections import deque

import param
import numpy as np
from holoviews import Dimension, NdLayout, GridSpace, Layout, NdOverlay
from holoviews.core.spaces import DynamicMap, HoloMap, Callable
from holoviews.core.options import Store
from holoviews.element import Image, Scatter, Curve, Text, Points
from holoviews.operation import histogram
from holoviews.plotting.util import initialize_dynamic
from holoviews.streams import Stream, LinkedStream, PointerXY, PointerX, PointerY, RangeX, Buffer, pointer_types
from holoviews.util import Dynamic
from holoviews.element.comparison import ComparisonTestCase

from ..utils import LoggingComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement

XY = Stream.define('XY', x=0,y=0)
X = Stream.define('X', x=0)
Y = Stream.define('Y', y=0)

frequencies =  np.linspace(0.5,2.0,5)
phases = np.linspace(0, np.pi*2, 5)
x,y = np.mgrid[-5:6, -5:6] * 0.1

def sine_array(phase, freq):
    return np.sin(phase + (freq*x**2+freq*y**2))

class ExampleParameterized(param.Parameterized):
    example = param.Number(default=1)

class DynamicMapConstructor(ComparisonTestCase):

    def test_simple_constructor_kdims(self):
        DynamicMap(lambda x: x, kdims=['test'])

    def test_simple_constructor_invalid_no_kdims(self):
        regexp = ("Callable '<lambda>' accepts more positional arguments than there are "
                  "kdims and stream parameters")
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(lambda x: x)

    def test_simple_constructor_invalid(self):
        regexp = (r"Callback '<lambda>' signature over \['x'\] does not accommodate "
                  r"required kdims \['x', 'y'\]")
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(lambda x: x, kdims=['x','y'])

    def test_simple_constructor_streams(self):
        DynamicMap(lambda x: x, streams=[PointerX()])

    def test_simple_constructor_streams_dict(self):
        pointerx = PointerX()
        DynamicMap(lambda x: x, streams=dict(x=pointerx.param.x))

    def test_simple_constructor_streams_dict_panel_widget(self):
        import panel
        DynamicMap(lambda x: x, streams=dict(x=panel.widgets.FloatSlider()))

    def test_simple_constructor_streams_dict_parameter(self):
        test = ExampleParameterized()
        DynamicMap(lambda x: x, streams=dict(x=test.param.example))

    def test_simple_constructor_streams_dict_class_parameter(self):
        DynamicMap(lambda x: x, streams=dict(x=ExampleParameterized.param.example))

    def test_simple_constructor_streams_dict_invalid(self):
        regexp = "Cannot handle value 3 in streams dictionary"
        with self.assertRaisesRegex(TypeError, regexp):
            DynamicMap(lambda x: x, streams=dict(x=3))

    def test_simple_constructor_streams_invalid_uninstantiated(self):
        regexp = ("The supplied streams list contains objects "
                  "that are not Stream instances:(.+?)")
        with self.assertRaisesRegex(TypeError, regexp):
            DynamicMap(lambda x: x, streams=[PointerX])

    def test_simple_constructor_streams_invalid_type(self):
        regexp = ("The supplied streams list contains objects "
                  "that are not Stream instances:(.+?)")
        with self.assertRaisesRegex(TypeError, regexp):
            DynamicMap(lambda x: x, streams=[3])

    def test_simple_constructor_streams_invalid_mismatch(self):
        regexp = "Callable '<lambda>' missing keywords to accept stream parameters: y"
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(lambda x: x, streams=[PointerXY()])

    def test_simple_constructor_positional_stream_args(self):
        DynamicMap(lambda v: v, streams=[PointerXY()], positional_stream_args=True)

    def test_simple_constructor_streams_invalid_mismatch_named(self):

        def foo(x): return x
        regexp = "Callable 'foo' missing keywords to accept stream parameters: y"
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(foo, streams=[PointerXY()])


class DynamicMapPositionalStreamArgs(ComparisonTestCase):
    def test_positional_stream_args_without_streams(self):
        fn = lambda i: Curve([i, i])
        dmap = DynamicMap(fn, kdims=['i'], positional_stream_args=True)
        self.assertEqual(dmap[0], Curve([0, 0]))

    def test_positional_stream_args_with_only_stream(self):
        fn = lambda s: Curve([s['x'], s['y']])
        xy_stream = XY(x=1, y=2)
        dmap = DynamicMap(fn, streams=[xy_stream], positional_stream_args=True)
        self.assertEqual(dmap[()], Curve([1, 2]))

        # Update stream values
        xy_stream.event(x=5, y=7)
        self.assertEqual(dmap[()], Curve([5, 7]))

    def test_positional_stream_args_with_single_kdim_and_stream(self):
        fn = lambda i, s: Points([i, i]) + Curve([s['x'], s['y']])
        xy_stream = XY(x=1, y=2)
        dmap = DynamicMap(
            fn, kdims=['i'], streams=[xy_stream], positional_stream_args=True
        )
        self.assertEqual(dmap[6], Points([6, 6]) + Curve([1, 2]))

        # Update stream values
        xy_stream.event(x=5, y=7)
        self.assertEqual(dmap[3], Points([3, 3]) + Curve([5, 7]))

    def test_positional_stream_args_with_multiple_kdims_and_stream(self):
        fn = lambda i, j, s1, s2: Points([i, j]) + Curve([s1['x'], s2['y']])
        x_stream = X(x=2)
        y_stream = Y(y=3)

        dmap = DynamicMap(
            fn,
            kdims=['i', 'j'],
            streams=[x_stream, y_stream],
            positional_stream_args=True
        )
        self.assertEqual(dmap[0, 1], Points([0, 1]) + Curve([2, 3]))

        # Update stream values
        x_stream.event(x=5)
        y_stream.event(y=6)
        self.assertEqual(dmap[3, 4], Points([3, 4]) + Curve([5, 6]))

    def test_initialize_with_overlapping_stream_params(self):
        fn = lambda xy0, xy1: \
             Points([xy0['x'], xy0['y']]) + Curve([xy1['x'], xy1['y']])
        xy_stream0 = XY(x=1, y=2)
        xy_stream1 = XY(x=3, y=4)
        dmap = DynamicMap(
            fn, streams=[xy_stream0, xy_stream1], positional_stream_args=True
        )
        self.assertEqual(dmap[()], Points([1, 2]) + Curve([3, 4]))


class DynamicMapMethods(ComparisonTestCase):

    def test_deep_relabel_label(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=['i']).relabel(label='Test')
        self.assertEqual(dmap[0].label, 'Test')

    def test_deep_relabel_group(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=['i']).relabel(group='Test')
        self.assertEqual(dmap[0].group, 'Test')

    def test_redim_dimension_name(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=['i']).redim(i='New')
        self.assertEqual(dmap.kdims[0].name, 'New')

    def test_redim_dimension_range_aux(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=['i']).redim.range(i=(0,1))
        self.assertEqual(dmap.kdims[0].range, (0,1))

    def test_redim_dimension_values_cache_reset_1D(self):
        # Setting the values should drop mismatching keys from the cache
        fn = lambda i: Curve([i,i])
        dmap = DynamicMap(fn, kdims=['i'])[{0,1,2,3,4,5}]
        self.assertEqual(dmap.keys(), [0,1,2,3,4,5])
        redimmed = dmap.redim.values(i=[2,3,5,6,8])
        self.assertEqual(redimmed.keys(), [2,3,5])

    def test_redim_dimension_values_cache_reset_2D_single(self):
        # Setting the values should drop mismatching keys from the cache
        fn = lambda i,j: Curve([i,j])
        keys = [(0,1),(1,0),(2,2),(2,5), (3,3)]
        dmap = DynamicMap(fn, kdims=['i','j'])[keys]
        self.assertEqual(dmap.keys(), keys)
        redimmed = dmap.redim.values(i=[2,10,50])
        self.assertEqual(redimmed.keys(), [(2,2),(2,5)])

    def test_redim_dimension_values_cache_reset_2D_multi(self):
        # Setting the values should drop mismatching keys from the cache
        fn = lambda i,j: Curve([i,j])
        keys = [(0,1),(1,0),(2,2),(2,5), (3,3)]
        dmap = DynamicMap(fn, kdims=['i','j'])[keys]
        self.assertEqual(dmap.keys(), keys)
        redimmed = dmap.redim.values(i=[2,10,50], j=[5,50,100])
        self.assertEqual(redimmed.keys(), [(2,5)])


    def test_redim_dimension_unit_aux(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=['i']).redim.unit(i='m/s')
        self.assertEqual(dmap.kdims[0].unit, 'm/s')

    def test_redim_dimension_type_aux(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=['i']).redim.type(i=int)
        self.assertEqual(dmap.kdims[0].type, int)

    def test_deep_redim_dimension_name(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=['i']).redim(x='X')
        self.assertEqual(dmap[0].kdims[0].name, 'X')

    def test_deep_redim_dimension_name_with_spec(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=['i']).redim(Image, x='X')
        self.assertEqual(dmap[0].kdims[0].name, 'X')

    def test_deep_getitem_bounded_kdims(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[:, 5:10][10], fn(10)[5:10])

    def test_deep_getitem_bounded_kdims_and_vdims(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[:, 5:10, 0:5][10], fn(10)[5:10, 0:5])

    def test_deep_getitem_cross_product_and_slice(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[[10, 11, 12], 5:10],
                         dmap.clone([(i, fn(i)[5:10]) for i in range(10, 13)]))

    def test_deep_getitem_index_and_slice(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[10, 5:10], fn(10)[5:10])

    def test_deep_getitem_cache_sliced(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap[10] # Add item to cache
        self.assertEqual(dmap[:, 5:10][10], fn(10)[5:10])

    def test_deep_select_slice_kdim(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap.select(x=(5, 10))[10], fn(10)[5:10])

    def test_deep_select_slice_kdim_and_vdims(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap.select(x=(5, 10), y=(0, 5))[10], fn(10)[5:10, 0:5])

    def test_deep_select_slice_kdim_no_match(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap.select(DynamicMap, x=(5, 10))[10], fn(10))

    def test_deep_apply_element_function(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(lambda x: x.clone(x.data*2))
        curve = fn(10)
        self.assertEqual(mapped[10], curve.clone(curve.data*2))

    def test_deep_apply_element_param_function(self):
        fn = lambda i: Curve(np.arange(i))
        class Test(param.Parameterized):
            a = param.Integer(default=1)
        test = Test()
        @param.depends(test.param.a)
        def op(obj, a):
            return obj.clone(obj.data*2)
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(op)
        test.a = 2
        curve = fn(10)
        self.assertEqual(mapped[10], curve.clone(curve.data*2))

    def test_deep_apply_element_function_with_kwarg(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(lambda x, label: x.relabel(label), label='New label')
        self.assertEqual(mapped[10], fn(10).relabel('New label'))

    def test_deep_map_apply_element_function_with_stream_kwarg(self):
        stream = Stream.define('Test', label='New label')()
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(lambda x, label: x.relabel(label), streams=[stream])
        self.assertEqual(mapped[10], fn(10).relabel('New label'))

    def test_deep_map_apply_parameterized_method_with_stream_kwarg(self):
        class Test(param.Parameterized):

            label = param.String(default='label')

            @param.depends('label')
            def value(self):
                return self.label.title()

        test = Test()
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(lambda x, label: x.relabel(label), label=test.value)
        curve = fn(10)
        self.assertEqual(mapped[10], curve.relabel('Label'))
        test.label = 'new label'
        self.assertEqual(mapped[10], curve.relabel('New Label'))

    def test_deep_apply_parameterized_method_with_dependency(self):
        class Test(param.Parameterized):

            label = param.String(default='label')

            @param.depends('label')
            def relabel(self, obj):
                return obj.relabel(self.label.title())

        test = Test()
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(test.relabel)
        curve = fn(10)
        self.assertEqual(mapped[10], curve.relabel('Label'))
        test.label = 'new label'
        self.assertEqual(mapped[10], curve.relabel('New Label'))

    def test_deep_apply_parameterized_method_with_dependency_and_static_kwarg(self):
        class Test(param.Parameterized):

            label = param.String(default='label')

            @param.depends('label')
            def relabel(self, obj, group):
                return obj.relabel(self.label.title(), group)

        test = Test()
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(test.relabel, group='Group')
        curve = fn(10)
        self.assertEqual(mapped[10], curve.relabel('Label', 'Group'))
        test.label = 'new label'
        self.assertEqual(mapped[10], curve.relabel('New Label', 'Group'))

    def test_deep_map_transform_element_type(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap[10]
        mapped = dmap.map(lambda x: Scatter(x), Curve)
        area = mapped[11]
        self.assertEqual(area, Scatter(fn(11)))

    def test_deep_apply_transform_element_type(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap[10]
        mapped = dmap.apply(lambda x: Scatter(x))
        area = mapped[11]
        self.assertEqual(area, Scatter(fn(11)))

    def test_deep_map_apply_dmap_function(self):
        fn = lambda i: Curve(np.arange(i))
        dmap1 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap2 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = (dmap1 + dmap2).map(lambda x: x[10], DynamicMap)
        self.assertEqual(mapped, Layout([('DynamicMap.I', fn(10)),
                                         ('DynamicMap.II', fn(10))]))

    def test_deep_map_apply_dmap_function_no_clone(self):
        fn = lambda i: Curve(np.arange(i))
        dmap1 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap2 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        layout = (dmap1 + dmap2)
        mapped = layout.map(lambda x: x[10], DynamicMap, clone=False)
        self.assertIs(mapped, layout)

    def test_dynamic_reindex_reorder(self):
        def history_callback(x, y, history=deque(maxlen=10)):
            history.append((x, y))
            return Points(list(history))
        dmap = DynamicMap(history_callback, kdims=['x', 'y'])
        reindexed = dmap.reindex(['y', 'x'])
        points = reindexed[2, 1]
        self.assertEqual(points, Points([(1, 2)]))

    def test_dynamic_reindex_drop_raises_exception(self):
        def history_callback(x, y, history=deque(maxlen=10)):
            history.append((x, y))
            return Points(list(history))
        dmap = DynamicMap(history_callback, kdims=['x', 'y'])
        exception = ("DynamicMap does not allow dropping dimensions, "
                     "reindex may only be used to reorder dimensions.")
        with self.assertRaisesRegex(ValueError, exception):
            dmap.reindex(['x'])

    def test_dynamic_groupby_kdims_and_streams(self):
        def plot_function(mydim, data):
            return Scatter(data[data[:, 2]==mydim])

        buff = Buffer(data=np.empty((0, 3)))
        dmap = DynamicMap(plot_function, streams=[buff], kdims='mydim').redim.values(mydim=[0, 1, 2])
        ndlayout = dmap.groupby('mydim', container_type=NdLayout)
        self.assertIsInstance(ndlayout[0], DynamicMap)
        data = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
        buff.send(data)
        self.assertIs(ndlayout[0].callback.inputs[0], dmap)
        self.assertIs(ndlayout[1].callback.inputs[0], dmap)
        self.assertIs(ndlayout[2].callback.inputs[0], dmap)
        self.assertEqual(ndlayout[0][()], Scatter([(0, 0)]))
        self.assertEqual(ndlayout[1][()], Scatter([(1, 1)]))
        self.assertEqual(ndlayout[2][()], Scatter([(2, 2)]))

    def test_dynamic_split_overlays_on_ndoverlay(self):
        dmap = DynamicMap(lambda: NdOverlay({i: Points([i]) for i in range(3)}))
        initialize_dynamic(dmap)
        keys, dmaps = dmap._split_overlays()
        self.assertEqual(keys, [(0,), (1,), (2,)])
        self.assertEqual(dmaps[0][()], Points([0]))
        self.assertEqual(dmaps[1][()], Points([1]))
        self.assertEqual(dmaps[2][()], Points([2]))

    def test_dynamic_split_overlays_on_overlay(self):
        dmap1 = DynamicMap(lambda: Points([]))
        dmap2 = DynamicMap(lambda: Curve([]))
        dmap = dmap1 * dmap2
        initialize_dynamic(dmap)
        keys, dmaps = dmap._split_overlays()
        self.assertEqual(keys, [('Points', 'I'), ('Curve', 'I')])
        self.assertEqual(dmaps[0][()], Points([]))
        self.assertEqual(dmaps[1][()], Curve([]))

    def test_dynamic_split_overlays_on_varying_order_overlay(self):
        def cb(i):
            if i%2 == 0:
                return Curve([]) * Points([])
            else:
                return Points([]) * Curve([])
        dmap = DynamicMap(cb, kdims='i').redim.range(i=(0, 4))
        initialize_dynamic(dmap)
        keys, dmaps = dmap._split_overlays()
        self.assertEqual(keys, [('Curve', 'I'), ('Points', 'I')])
        self.assertEqual(dmaps[0][0], Curve([]))
        self.assertEqual(dmaps[0][1], Curve([]))
        self.assertEqual(dmaps[1][0], Points([]))
        self.assertEqual(dmaps[1][1], Points([]))

    def test_dynamic_split_overlays_on_missing_item_in_overlay(self):
        def cb(i):
            if i%2 == 0:
                return Curve([]) * Points([])
            else:
                return Scatter([]) * Curve([])
        dmap = DynamicMap(cb, kdims='i').redim.range(i=(0, 4))
        initialize_dynamic(dmap)
        keys, dmaps = dmap._split_overlays()
        self.assertEqual(keys, [('Curve', 'I'), ('Points', 'I')])
        self.assertEqual(dmaps[0][0], Curve([]))
        self.assertEqual(dmaps[0][1], Curve([]))
        self.assertEqual(dmaps[1][0], Points([]))
        with self.assertRaises(KeyError):
            dmaps[1][1]



class DynamicMapOptionsTests(CustomBackendTestCase):

    def test_dynamic_options(self):
        dmap = DynamicMap(lambda X: ExampleElement(None), kdims=['X']).redim.range(X=(0,10))
        dmap = dmap.options(plot_opt1='red')
        opts = Store.lookup_options('backend_1', dmap[0], 'plot')
        self.assertEqual(opts.options, {'plot_opt1': 'red'})

    def test_dynamic_options_no_clone(self):
        dmap = DynamicMap(lambda X: ExampleElement(None), kdims=['X']).redim.range(X=(0,10))
        dmap.options(plot_opt1='red', clone=False)
        opts = Store.lookup_options('backend_1', dmap[0], 'plot')
        self.assertEqual(opts.options, {'plot_opt1': 'red'})

    def test_dynamic_opts_link_inputs(self):
        stream = LinkedStream()
        inputs = [DynamicMap(lambda: None, streams=[stream])]
        dmap = DynamicMap(Callable(lambda X: ExampleElement(None), inputs=inputs),
                          kdims=['X']).redim.range(X=(0,10))
        styled_dmap = dmap.options(plot_opt1='red', clone=False)
        opts = Store.lookup_options('backend_1', dmap[0], 'plot')
        self.assertEqual(opts.options, {'plot_opt1': 'red'})
        self.assertIs(styled_dmap, dmap)
        self.assertTrue(dmap.callback.link_inputs)
        unstyled_dmap = dmap.callback.inputs[0].callback.inputs[0]
        opts = Store.lookup_options('backend_1', unstyled_dmap[0], 'plot')
        self.assertEqual(opts.options, {})
        original_dmap = unstyled_dmap.callback.inputs[0]
        self.assertIs(stream, original_dmap.streams[0])


class DynamicMapUnboundedProperty(ComparisonTestCase):

    def test_callable_bounded_init(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=[Dimension('dim', range=(0,10))])
        self.assertEqual(dmap.unbounded, [])

    def test_callable_bounded_clone(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=[Dimension('dim', range=(0,10))])
        self.assertEqual(dmap, dmap.clone())
        self.assertEqual(dmap.unbounded, [])

    def test_sampled_unbounded_init(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        self.assertEqual(dmap.unbounded, ['i'])

    def test_sampled_unbounded_resample(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        self.assertEqual(dmap[{0, 1, 2}].keys(), [0, 1, 2])
        self.assertEqual(dmap.unbounded, ['i'])

    def test_mixed_kdim_streams_unbounded(self):
        dmap=DynamicMap(lambda x,y,z: x+y, kdims=['z'], streams=[XY()])
        self.assertEqual(dmap.unbounded, ['z'])

    def test_mixed_kdim_streams_bounded_redim(self):
        dmap=DynamicMap(lambda x,y,z: x+y, kdims=['z'], streams=[XY()])
        self.assertEqual(dmap.redim.range(z=(-0.5,0.5)).unbounded, [])


class DynamicMapCurrentKeyProperty(ComparisonTestCase):

    def test_current_key_None_on_init(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=[Dimension('dim', range=(0,10))])
        self.assertIsNone(dmap.current_key)

    def test_current_key_one_dimension(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=[Dimension('dim', range=(0,10))])
        dmap[0]
        self.assertEqual(dmap.current_key, 0)
        dmap[1]
        self.assertEqual(dmap.current_key, 1)
        dmap[0]
        self.assertEqual(dmap.current_key, 0)
        self.assertNotEqual(dmap.current_key, dmap.last_key)

    def test_current_key_multiple_dimensions(self):
        fn = lambda i, j: Curve([i, j])
        dmap=DynamicMap(fn, kdims=[Dimension('i', range=(0,5)), Dimension('j', range=(0,5))])
        dmap[0, 2]
        self.assertEqual(dmap.current_key, (0, 2))
        dmap[5, 5]
        self.assertEqual(dmap.current_key, (5, 5))
        dmap[0, 2]
        self.assertEqual(dmap.current_key, (0, 2))
        self.assertNotEqual(dmap.current_key, dmap.last_key)


class DynamicTransferStreams(ComparisonTestCase):

    def setUp(self):
        self.dimstream = PointerX(x=0)
        self.stream = PointerY(y=0)
        self.dmap = DynamicMap(lambda x, y, z: Curve([x, y, z]),
                               kdims=['x', 'z'], streams=[self.stream, self.dimstream])

    def test_dynamic_redim_inherits_streams(self):
        redimmed = self.dmap.redim.range(z=(0, 5))
        self.assertEqual(redimmed.streams, self.dmap.streams)

    def test_dynamic_relabel_inherits_streams(self):
        relabelled = self.dmap.relabel(label='Test')
        self.assertEqual(relabelled.streams, self.dmap.streams)

    def test_dynamic_map_inherits_streams(self):
        mapped = self.dmap.map(lambda x: x, Curve)
        self.assertEqual(mapped.streams, self.dmap.streams)

    def test_dynamic_select_inherits_streams(self):
        selected = self.dmap.select(Curve, x=(0, 5))
        self.assertEqual(selected.streams, self.dmap.streams)

    def test_dynamic_hist_inherits_streams(self):
        hist = self.dmap.hist(adjoin=False)
        self.assertEqual(hist.streams, self.dmap.streams)

    def test_dynamic_mul_inherits_dim_streams(self):
        hist = self.dmap * self.dmap
        self.assertEqual(hist.streams, self.dmap.streams[1:])

    def test_dynamic_util_inherits_dim_streams(self):
        hist = Dynamic(self.dmap)
        self.assertEqual(hist.streams, self.dmap.streams[1:])

    def test_dynamic_util_parameterized_method(self):
        class Test(param.Parameterized):
            label = param.String(default='test')

            @param.depends('label')
            def apply_label(self, obj):
                return obj.relabel(self.label)

        test = Test()
        dmap = Dynamic(self.dmap, operation=test.apply_label)
        test.label = 'custom label'
        self.assertEqual(dmap[(0, 3)].label, 'custom label')

    def test_dynamic_util_inherits_dim_streams_clash(self):
        exception = (r"The supplied stream objects PointerX\(x=None\) and "
                     r"PointerX\(x=0\) clash on the following parameters: \['x'\]")
        with self.assertRaisesRegex(Exception, exception):
            Dynamic(self.dmap, streams=[PointerX])

    def test_dynamic_util_inherits_dim_streams_clash_dict(self):
        exception = (r"The supplied stream objects PointerX\(x=None\) and "
                     r"PointerX\(x=0\) clash on the following parameters: \['x'\]")
        with self.assertRaisesRegex(Exception, exception):
            Dynamic(self.dmap, streams=dict(x=PointerX.param.x))



class DynamicTestOperation(ComparisonTestCase):

    def test_dynamic_operation(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        dmap_with_fn = Dynamic(dmap, operation=lambda x: x.clone(x.data*2))
        self.assertEqual(dmap_with_fn[5], Image(sine_array(0,5)*2))

    def test_dynamic_operation_on_hmap(self):
        hmap = HoloMap({i: Image(sine_array(0,i)) for i in range(10)})
        dmap = Dynamic(hmap, operation=lambda x: x)
        self.assertEqual(dmap.kdims[0].name, hmap.kdims[0].name)
        self.assertEqual(dmap.kdims[0].values, hmap.keys())

    def test_dynamic_operation_link_inputs_not_transferred_on_clone(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        dmap_with_fn = Dynamic(dmap, link_inputs=False, operation=lambda x: x.clone(x.data*2))
        self.assertTrue(dmap_with_fn.clone().callback.link_inputs)

    def test_dynamic_operation_on_element(self):
        img = Image(sine_array(0,5))
        posxy = PointerXY(x=2, y=1)
        dmap_with_fn = Dynamic(img, operation=lambda obj, x, y: obj.clone(obj.data*x+y),
                               streams=[posxy])
        element = dmap_with_fn[()]
        self.assertEqual(element, Image(sine_array(0,5)*2+1))
        self.assertEqual(dmap_with_fn.streams, [posxy])

    def test_dynamic_operation_on_element_dict(self):
        img = Image(sine_array(0,5))
        posxy = PointerXY(x=3, y=1)
        dmap_with_fn = Dynamic(img, operation=lambda obj, x, y: obj.clone(obj.data*x+y),
                               streams=dict(x=posxy.param.x, y=posxy.param.y))
        element = dmap_with_fn[()]
        self.assertEqual(element, Image(sine_array(0,5)*3+1))

    def test_dynamic_operation_with_kwargs(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        def fn(x, multiplier=2):
            return x.clone(x.data*multiplier)
        dmap_with_fn = Dynamic(dmap, operation=fn, kwargs=dict(multiplier=3))
        self.assertEqual(dmap_with_fn[5], Image(sine_array(0,5)*3))

    def test_dynamic_operation_init_renamed_stream_params(self):
        img = Image(sine_array(0,5))
        stream = RangeX(rename={'x_range': 'bin_range'})
        histogram(img, bin_range=(0, 1), streams=[stream], dynamic=True)
        self.assertEqual(stream.x_range, (0, 1))

    def test_dynamic_operation_init_stream_params(self):
        img = Image(sine_array(0,5))
        stream = Stream.define('TestStream', bin_range=None)()
        histogram(img, bin_range=(0, 1), streams=[stream], dynamic=True)
        self.assertEqual(stream.bin_range, (0, 1))




class DynamicTestOverlay(ComparisonTestCase):

    def test_dynamic_element_overlay(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        dynamic_overlay = dmap * Image(sine_array(0,10))
        overlaid = Image(sine_array(0,5)) * Image(sine_array(0,10))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_element_underlay(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        dynamic_overlay = Image(sine_array(0,10)) * dmap
        overlaid = Image(sine_array(0,10)) * Image(sine_array(0,5))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_dynamicmap_overlay(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        fn2 = lambda i: Image(sine_array(0,i*2))
        dmap2=DynamicMap(fn2, kdims=['i'])
        dynamic_overlay = dmap * dmap2
        overlaid = Image(sine_array(0,5)) * Image(sine_array(0,10))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_holomap_overlay(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, kdims=['i'])
        hmap = HoloMap({i: Image(sine_array(0,i*2)) for i in range(10)}, kdims=['i'])
        dynamic_overlay = dmap * hmap
        overlaid = Image(sine_array(0,5)) * Image(sine_array(0,10))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_overlay_memoization(self):
        """Tests that Callable memoizes unchanged callbacks"""
        def fn(x, y):
            return Scatter([(x, y)])
        dmap = DynamicMap(fn, kdims=[], streams=[PointerXY()])

        counter = [0]
        def fn2(x, y):
            counter[0] += 1
            return Image(np.random.rand(10, 10))
        dmap2 = DynamicMap(fn2, kdims=[], streams=[PointerXY()])

        overlaid = dmap * dmap2
        overlay = overlaid[()]
        self.assertEqual(overlay.Scatter.I, fn(0, 0))

        dmap.event(x=1, y=2)
        overlay = overlaid[()]
        # Ensure dmap return value was updated
        self.assertEqual(overlay.Scatter.I, fn(1, 2))
        # Ensure dmap2 callback was called only once
        self.assertEqual(counter[0], 1)

    def test_dynamic_event_renaming_valid(self):

        def fn(x1, y1):
            return Scatter([(x1, y1)])

        xy = PointerXY(rename={'x':'x1','y':'y1'})
        dmap = DynamicMap(fn, kdims=[], streams=[xy])
        dmap.event(x1=1, y1=2)

    def test_dynamic_event_renaming_invalid(self):
        def fn(x1, y1):
            return Scatter([(x1, y1)])

        xy = PointerXY(rename={'x':'x1','y':'y1'})
        dmap = DynamicMap(fn, kdims=[], streams=[xy])

        regexp = '(.+?)do not correspond to stream parameters'
        with self.assertRaisesRegex(KeyError, regexp):
            dmap.event(x=1, y=2)


class DynamicCallableMemoize(ComparisonTestCase):

    def test_dynamic_keydim_not_memoize(self):
        dmap = DynamicMap(lambda x: Curve([(0, x)]), kdims=['x'])
        self.assertEqual(dmap[0], Curve([(0, 0)]))
        self.assertEqual(dmap[1], Curve([(0, 1)]))

    def test_dynamic_keydim_memoize(self):
        dmap = DynamicMap(lambda x: Curve([(0, x)]), kdims=['x'])
        self.assertIs(dmap[0], dmap[0])

    def test_dynamic_keydim_memoize_disable(self):
        dmap = DynamicMap(Callable(lambda x: Curve([(0, x)]), memoize=False), kdims=['x'])
        first = dmap[0]
        del dmap.data[(0,)]
        second = dmap[0]
        self.assertIsNot(first, second)

    def test_dynamic_callable_memoize(self):
        # Always memoized only one of each held
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))

        x = PointerX()
        dmap = DynamicMap(history_callback, kdims=[], streams=[x])

        # Add stream subscriber mocking plot
        x.add_subscriber(lambda **kwargs: dmap[()])

        for i in range(2):
            x.event(x=1)

        self.assertEqual(dmap[()], Curve([1]))

        for i in range(2):
            x.event(x=2)

        self.assertEqual(dmap[()], Curve([1, 2]))


    def test_dynamic_callable_disable_callable_memoize(self):
        # Disabling Callable.memoize means no memoization is applied,
        # every access to DynamicMap calls callback and adds sample
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))

        x = PointerX()
        callable_obj = Callable(history_callback, memoize=False)
        dmap = DynamicMap(callable_obj, kdims=[], streams=[x])

        # Add stream subscriber mocking plot
        x.add_subscriber(lambda **kwargs: dmap[()])

        for i in range(2):
            x.event(x=1)
        self.assertEqual(dmap[()], Curve([1, 1, 1]))

        for i in range(2):
            x.event(x=2)
        self.assertEqual(dmap[()], Curve([1, 1, 1, 2, 2, 2]))


class StreamSubscribersAddandClear(ComparisonTestCase):

    def setUp(self):
        self.fn1 = lambda x: x
        self.fn2 = lambda x: x**2
        self.fn3 = lambda x: x**3
        self.fn4 = lambda x: x**4

    def test_subscriber_clear_all(self):
        pointerx = PointerX(x=2)
        pointerx.add_subscriber(self.fn1, precedence=0)
        pointerx.add_subscriber(self.fn2, precedence=1)
        pointerx.add_subscriber(self.fn3, precedence=1.5)
        pointerx.add_subscriber(self.fn4, precedence=10)
        self.assertEqual(pointerx.subscribers,  [self.fn1,self.fn2,self.fn3,self.fn4])
        pointerx.clear('all')
        self.assertEqual(pointerx.subscribers,  [])

    def test_subscriber_clear_user(self):
        pointerx = PointerX(x=2)
        pointerx.add_subscriber(self.fn1, precedence=0)
        pointerx.add_subscriber(self.fn2, precedence=1)
        pointerx.add_subscriber(self.fn3, precedence=1.5)
        pointerx.add_subscriber(self.fn4, precedence=10)
        self.assertEqual(pointerx.subscribers,  [self.fn1,self.fn2,self.fn3,self.fn4])
        pointerx.clear('user')
        self.assertEqual(pointerx.subscribers,  [self.fn3,self.fn4])


    def test_subscriber_clear_internal(self):
        pointerx = PointerX(x=2)
        pointerx.add_subscriber(self.fn1, precedence=0)
        pointerx.add_subscriber(self.fn2, precedence=1)
        pointerx.add_subscriber(self.fn3, precedence=1.5)
        pointerx.add_subscriber(self.fn4, precedence=10)
        self.assertEqual(pointerx.subscribers,  [self.fn1,self.fn2,self.fn3,self.fn4])
        pointerx.clear('internal')
        self.assertEqual(pointerx.subscribers,  [self.fn1,self.fn2])


class DynamicStreamReset(ComparisonTestCase):

    def test_dynamic_callable_stream_transient(self):
        # Enable transient stream meaning memoization only happens when
        # stream is inactive, should have sample for each call to
        # stream.update
        def history_callback(x, history=deque(maxlen=10)):
            if x is not None:
                history.append(x)
            return Curve(list(history))

        x = PointerX(transient=True)
        dmap = DynamicMap(history_callback, kdims=[], streams=[x])

        # Add stream subscriber mocking plot
        x.add_subscriber(lambda **kwargs: dmap[()])

        for i in range(2):
            x.event(x=1)
        self.assertEqual(dmap[()], Curve([1, 1]))

        for i in range(2):
            x.event(x=2)

        self.assertEqual(dmap[()], Curve([1, 1, 2, 2]))

    def test_dynamic_stream_transients(self):
        # Ensure Stream reset option resets streams to default value
        # when not triggering
        global xresets, yresets
        xresets, yresets = 0, 0
        def history_callback(x, y, history=deque(maxlen=10)):
            global xresets, yresets
            if x is None:
                xresets += 1
            else:
                history.append(x)
            if y is None:
                yresets += 1

            return Curve(list(history))

        x = PointerX(transient=True)
        y = PointerY(transient=True)
        dmap = DynamicMap(history_callback, kdims=[], streams=[x, y])

        # Add stream subscriber mocking plot
        x.add_subscriber(lambda **kwargs: dmap[()])
        y.add_subscriber(lambda **kwargs: dmap[()])

        # Update each stream and count when None default appears
        for i in range(2):
            x.event(x=i)
            y.event(y=i)

        self.assertEqual(xresets, 2)
        self.assertEqual(yresets, 2)

    def test_dynamic_callable_stream_hashkey(self):
        # Enable transient stream meaning memoization only happens when
        # stream is inactive, should have sample for each call to
        # stream.update
        def history_callback(x, history=deque(maxlen=10)):
            if x is not None:
                history.append(x)
            return Curve(list(history))

        class NoMemoize(PointerX):
            x = param.ClassSelector(class_=pointer_types, default=None, constant=True)
            @property
            def hashkey(self): return {'hash': uuid.uuid4().hex}

        x = NoMemoize()
        dmap = DynamicMap(history_callback, kdims=[], streams=[x])

        # Add stream subscriber mocking plot
        x.add_subscriber(lambda **kwargs: dmap[()])

        for i in range(2):
            x.event(x=1)
        self.assertEqual(dmap[()], Curve([1, 1, 1]))

        for i in range(2):
            x.event(x=2)

        self.assertEqual(dmap[()], Curve([1, 1, 1, 2, 2, 2]))



class TestPeriodicStreamUpdate(ComparisonTestCase):

    def test_periodic_counter_blocking(self):
        class Counter(object):
            def __init__(self):
                self.count = 0
            def __call__(self):
                self.count += 1
                return Curve([1,2,3])

        next_stream = Stream.define('Next')()
        counter = Counter()
        dmap = DynamicMap(counter, streams=[next_stream])
        # Add stream subscriber mocking plot
        next_stream.add_subscriber(lambda **kwargs: dmap[()])
        dmap.periodic(0.01, 100)
        self.assertEqual(counter.count, 100)

    def test_periodic_param_fn_blocking(self):
        def callback(x): return Curve([1,2,3])
        xval = Stream.define('x',x=0)()
        dmap = DynamicMap(callback, streams=[xval])
        # Add stream subscriber mocking plot
        xval.add_subscriber(lambda **kwargs: dmap[()])
        dmap.periodic(0.01, 100, param_fn=lambda i: {'x':i})
        self.assertEqual(xval.x, 100)

    def test_periodic_param_fn_non_blocking(self):
        def callback(x): return Curve([1,2,3])
        xval = Stream.define('x',x=0)()
        dmap = DynamicMap(callback, streams=[xval])
        # Add stream subscriber mocking plot
        xval.add_subscriber(lambda **kwargs: dmap[()])

        self.assertNotEqual(xval.x, 100)
        dmap.periodic(0.0001, 100, param_fn=lambda i: {'x': i}, block=False)
        time.sleep(2)
        if not dmap.periodic.instance.completed:
            raise RuntimeError('Periodic callback timed out.')
        dmap.periodic.stop()
        self.assertEqual(xval.x, 100)

    def test_periodic_param_fn_blocking_period(self):
        def callback(x):
            return Curve([1,2,3])
        xval = Stream.define('x',x=0)()
        dmap = DynamicMap(callback, streams=[xval])
        # Add stream subscriber mocking plot
        xval.add_subscriber(lambda **kwargs: dmap[()])
        start = time.time()
        dmap.periodic(0.5, 10, param_fn=lambda i: {'x':i}, block=True)
        end = time.time()
        self.assertEqual((end - start) > 5, True)


    def test_periodic_param_fn_blocking_timeout(self):
        def callback(x):
            return Curve([1,2,3])
        xval = Stream.define('x',x=0)()
        dmap = DynamicMap(callback, streams=[xval])
        # Add stream subscriber mocking plot
        xval.add_subscriber(lambda **kwargs: dmap[()])
        start = time.time()
        dmap.periodic(0.5, 100, param_fn=lambda i: {'x':i}, timeout=3)
        end = time.time()
        self.assertEqual((end - start) < 5, True)


class DynamicCollate(LoggingComparisonTestCase):

    def test_dynamic_collate_layout(self):
        def callback():
            return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
        dmap = DynamicMap(callback, kdims=[])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [('Image', 'I'), ('Text', 'I')])
        self.assertEqual(layout.Image.I[()], Image(np.array([[0, 1], [2, 3]])))

    def test_dynamic_collate_layout_raise_no_remapping_error(self):
        def callback(x, y):
            return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
        stream = PointerXY()
        cb_callable = Callable(callback)
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        with self.assertRaisesRegex(ValueError, 'The following streams are set to be automatically linked'):
            dmap.collate()

    def test_dynamic_collate_layout_raise_ambiguous_remapping_error(self):
        def callback(x, y):
            return Image(np.array([[0, 1], [2, 3]])) + Image(np.array([[0, 1], [2, 3]]))
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={'Image': [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        with self.assertRaisesRegex(ValueError, 'The stream_mapping supplied on the Callable is ambiguous'):
            dmap.collate()

    def test_dynamic_collate_layout_with_integer_stream_mapping(self):
        def callback(x, y):
            return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={0: [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [('Image', 'I'), ('Text', 'I')])
        self.assertIs(stream.source, layout.Image.I)

    def test_dynamic_collate_layout_with_spec_stream_mapping(self):
        def callback(x, y):
            return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={'Image': [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [('Image', 'I'), ('Text', 'I')])
        self.assertIs(stream.source, layout.Image.I)

    def test_dynamic_collate_ndlayout(self):
        def callback():
            return NdLayout({i: Image(np.array([[i, 1], [2, 3]])) for i in range(1, 3)})
        dmap = DynamicMap(callback, kdims=[])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [1, 2])
        self.assertEqual(layout[1][()], Image(np.array([[1, 1], [2, 3]])))

    def test_dynamic_collate_ndlayout_with_integer_stream_mapping(self):
        def callback(x, y):
            return NdLayout({i: Image(np.array([[i, 1], [2, 3]])) for i in range(1, 3)})
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={0: [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [1, 2])
        self.assertIs(stream.source, layout[1])

    def test_dynamic_collate_ndlayout_with_key_stream_mapping(self):
        def callback(x, y):
            return NdLayout({i: Image(np.array([[i, 1], [2, 3]])) for i in range(1, 3)})
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={(1,): [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [1, 2])
        self.assertIs(stream.source, layout[1])

    def test_dynamic_collate_grid(self):
        def callback():
            return GridSpace({(i, j): Image(np.array([[i, j], [2, 3]]))
                              for i in range(1, 3) for j in range(1, 3)})
        dmap = DynamicMap(callback, kdims=[])
        grid = dmap.collate()
        self.assertEqual(list(grid.keys()), [(i, j) for i in range(1, 3)
                                             for j in range(1, 3)])
        self.assertEqual(grid[(0, 1)][()], Image(np.array([[1, 1], [2, 3]])))

    def test_dynamic_collate_grid_with_integer_stream_mapping(self):
        def callback():
            return GridSpace({(i, j): Image(np.array([[i, j], [2, 3]]))
                              for i in range(1, 3) for j in range(1, 3)})
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={1: [stream]})
        dmap = DynamicMap(cb_callable, kdims=[])
        grid = dmap.collate()
        self.assertEqual(list(grid.keys()), [(i, j) for i in range(1, 3)
                                             for j in range(1, 3)])
        self.assertEqual(stream.source, grid[(1, 2)])

    def test_dynamic_collate_grid_with_key_stream_mapping(self):
        def callback():
            return GridSpace({(i, j): Image(np.array([[i, j], [2, 3]]))
                              for i in range(1, 3) for j in range(1, 3)})
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={(1, 2): [stream]})
        dmap = DynamicMap(cb_callable, kdims=[])
        grid = dmap.collate()
        self.assertEqual(list(grid.keys()), [(i, j) for i in range(1, 3)
                                             for j in range(1, 3)])
        self.assertEqual(stream.source, grid[(1, 2)])

    def test_dynamic_collate_layout_with_changing_label(self):
        def callback(i):
            return Layout([Curve([], label=str(j)) for j in range(i, i+2)])
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(0, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        el1, el2 = dmap1[2], dmap2[2]
        self.assertEqual(el1.label, '2')
        self.assertEqual(el2.label, '3')

    def test_dynamic_collate_ndlayout_with_changing_keys(self):
        def callback(i):
            return NdLayout({j: Curve([], label=str(j)) for j in range(i, i+2)})
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(0, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        el1, el2 = dmap1[2], dmap2[2]
        self.assertEqual(el1.label, '2')
        self.assertEqual(el2.label, '3')

    def test_dynamic_collate_gridspace_with_changing_keys(self):
        def callback(i):
            return GridSpace({j: Curve([], label=str(j)) for j in range(i, i+2)}, 'X')
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(0, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        el1, el2 = dmap1[2], dmap2[2]
        self.assertEqual(el1.label, '2')
        self.assertEqual(el2.label, '3')

    def test_dynamic_collate_gridspace_with_changing_items_raises(self):
        def callback(i):
            return GridSpace({j: Curve([], label=str(j)) for j in range(i)}, 'X')
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(2, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        err = 'Collated DynamicMaps must return GridSpace with consistent number of items.'
        with self.assertRaisesRegex(ValueError, err):
            dmap1[4]
        self.log_handler.assertContains('WARNING', err)

    def test_dynamic_collate_gridspace_with_changing_item_types_raises(self):
        def callback(i):
            eltype = Image if i%2 else Curve
            return GridSpace({j: eltype([], label=str(j)) for j in range(i, i+2)}, 'X')
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(2, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        err = ('The objects in a GridSpace returned by a DynamicMap must '
               'consistently return the same number of items of the same type.')
        with self.assertRaisesRegex(ValueError, err):
            dmap1[3]
        self.log_handler.assertContains('WARNING', err)
