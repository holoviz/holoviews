from collections import deque

import numpy as np
from holoviews import Dimension, NdLayout, GridSpace, Layout
from holoviews.core.spaces import DynamicMap, HoloMap, Callable
from holoviews.element import Image, Scatter, Curve, Text, Points
from holoviews.streams import PointerXY, PointerX, PointerY
from holoviews.util import Dynamic
from holoviews.element.comparison import ComparisonTestCase

frequencies =  np.linspace(0.5,2.0,5)
phases = np.linspace(0, np.pi*2, 5)
x,y = np.mgrid[-5:6, -5:6] * 0.1

def sine_array(phase, freq):
    return np.sin(phase + (freq*x**2+freq*y**2))



class DynamicMapConstructor(ComparisonTestCase):

    def test_simple_constructor_kdims(self):
        DynamicMap(lambda x: x, kdims=['test'])

    def test_simple_constructor_invalid_no_kdims(self):
        regexp = ('Callable accepts more positional arguments than there are '
                  'kdims and stream parameters')
        with self.assertRaisesRegexp(KeyError, regexp):
            DynamicMap(lambda x: x)

    def test_simple_constructor_invalid(self):
        regexp = ("Callback signature over \['x'\] does not accommodate "
                  "required kdims \['x', 'y'\]")
        with self.assertRaisesRegexp(KeyError, regexp):
            DynamicMap(lambda x: x, kdims=['x','y'])

    def test_simple_constructor_streams(self):
        DynamicMap(lambda x: x, streams=[PointerX()])

    def test_simple_constructor_streams_invalid_uninstantiated(self):
        regexp = ("The supplied streams list contains objects "
                  "that are not Stream instances:(.+?)")
        with self.assertRaisesRegexp(TypeError, regexp):
            DynamicMap(lambda x: x, streams=[PointerX])

    def test_simple_constructor_streams_invalid_type(self):
        regexp = ("The supplied streams list contains objects "
                  "that are not Stream instances:(.+?)")
        with self.assertRaisesRegexp(TypeError, regexp):
            DynamicMap(lambda x: x, streams=[3])

    def test_simple_constructor_streams_invalid_mismatch(self):
        regexp = 'Callable missing keywords to accept y stream parameters'
        with self.assertRaisesRegexp(KeyError, regexp):
            DynamicMap(lambda x: x, streams=[PointerXY()])



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

    def test_deep_map_apply_element_function(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.map(lambda x: x.clone(x.data*2), Curve)
        curve = fn(10)
        self.assertEqual(mapped[10], curve.clone(curve.data*2))

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
        with self.assertRaisesRegexp(ValueError, exception):
            reindexed = dmap.reindex(['x'])


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

    def test_sampled_bounded_resample(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        self.assertEqual(dmap[{0, 1, 2}].keys(), [0, 1, 2])
        self.assertEqual(dmap.unbounded, ['i'])


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

    def test_dynamic_util_inherits_dim_streams_clash(self):
        exception = ("The supplied stream objects PointerX\(x=None\) and "
                     "PointerX\(x=0\) clash on the following parameters: \['x'\]")
        with self.assertRaisesRegexp(Exception, exception):
            hist = Dynamic(self.dmap, streams=[PointerX])



class DynamicTestOperation(ComparisonTestCase):

    def test_dynamic_operation(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        dmap_with_fn = Dynamic(dmap, operation=lambda x: x.clone(x.data*2))
        self.assertEqual(dmap_with_fn[5], Image(sine_array(0,5)*2))


    def test_dynamic_operation_with_kwargs(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=['i'])
        def fn(x, multiplier=2):
            return x.clone(x.data*multiplier)
        dmap_with_fn = Dynamic(dmap, operation=fn, kwargs=dict(multiplier=3))
        self.assertEqual(dmap_with_fn[5], Image(sine_array(0,5)*3))



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
        with self.assertRaisesRegexp(KeyError, regexp):
            dmap.event(x=1, y=2)


class DynamicCallableMemoize(ComparisonTestCase):

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
            x.update(x=1)

        self.assertEqual(dmap[()], Curve([1]))

        for i in range(2):
            x.update(x=2)

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
            x.update(x=1)
        self.assertEqual(dmap[()], Curve([1, 1, 1]))

        for i in range(2):
            x.update(x=2)
        self.assertEqual(dmap[()], Curve([1, 1, 1, 2, 2, 2]))



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
            x.update(x=1)
        self.assertEqual(dmap[()], Curve([1, 1]))

        for i in range(2):
            x.update(x=2)

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
            x.update(x=i)
            y.update(y=i)

        self.assertEqual(xresets, 2)
        self.assertEqual(yresets, 2)


class DynamicCollate(ComparisonTestCase):

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
        with self.assertRaisesRegexp(ValueError, 'The following streams are set to be automatically linked'):
            layout = dmap.collate()

    def test_dynamic_collate_layout_raise_ambiguous_remapping_error(self):
        def callback(x, y):
            return Image(np.array([[0, 1], [2, 3]])) + Image(np.array([[0, 1], [2, 3]]))
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={'Image': [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        with self.assertRaisesRegexp(ValueError, 'The stream_mapping supplied on the Callable is ambiguous'):
            layout = dmap.collate()

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
